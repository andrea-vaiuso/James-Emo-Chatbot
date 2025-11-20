from transformers import pipeline

class EmotionalStateManager:
    def __init__(self, emotional_model_name="ayoubkirouane/BERT-Emotions-Classifier", max_reasons_per_emotion=10,
                 reason_sumup_model=None, reason_sumup_tokenizer=None):
        classifier = pipeline("text-classification", model=emotional_model_name, top_k=None)
        self.classifier = classifier
        self.labels = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
        self.max_reasons_per_emotion = max_reasons_per_emotion
        self.reason_sumup_model = reason_sumup_model
        self.reason_sumup_tokenizer = reason_sumup_tokenizer

    def analyze_emotional_state(self, text: str) -> dict:
        results = self.classifier(text)
        if results and isinstance(results[0], list):
            flat_results = [r for sub in results for r in sub]
        else:
            flat_results = results

        emotion_scores = {label: 0.0 for label in self.labels}
        for result in flat_results:
            label = result["label"].lower()
            score = float(result["score"])
            if label in emotion_scores:
                emotion_scores[label] += score


        total = sum(emotion_scores.values())
        if total > 0:
            for label in emotion_scores:
                emotion_scores[label] /= total

        # Wrap into value+reasons structure
        emotion_struct = {
            label: {"value": score, "reasons": []}
            for label, score in emotion_scores.items()
        }

        # Add the reason (text) to the dominant emotion
        dominant_emotion = max(emotion_struct, key=lambda k: emotion_struct[k]["value"])
        if self.reason_sumup_model is not None and self.reason_sumup_tokenizer is not None:
            summary_reason = self._summarize_reasons([text], dominant_emotion, max_new_tokens=10)
        else:
            summary_reason = text
        emotion_struct[dominant_emotion]["reasons"].append(summary_reason)
        if len(emotion_struct[dominant_emotion]["reasons"]) > self.max_reasons_per_emotion:
            emotion_struct[dominant_emotion]["reasons"] = \
                emotion_struct[dominant_emotion]["reasons"][-self.max_reasons_per_emotion:]

        return emotion_struct

    def _extract_value_and_reasons(self, entry):
        """Helper: accept either float or {'value': float, 'reasons': [...]}. """
        if isinstance(entry, dict):
            return float(entry.get("value", 0.0)), list(entry.get("reasons", []))
        else:
            return float(entry), []
        
    def _summarize_reasons(self, reasons: list, emotion:str, max_new_tokens=10) -> str:
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are an AI language model tasked with summarizing reasons for feeling {emotion}.\n"
            f"Summarize the following reasons into a concise phrase of maximum {max_new_tokens} tokens:\n"
            f"Example format: 'I feel {emotion} because ...'\n"
            "Reasons:\n"
            f"{reasons}"
            "<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"Summary: "
        )
        inputs = self.reason_sumup_tokenizer(prompt, return_tensors="pt").to(self.reason_sumup_model.device)
        gen_ids = self.reason_sumup_model.generate(inputs.input_ids, max_new_tokens=max_new_tokens)
        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        summary = self.reason_sumup_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return summary

    def update_affect_state(
            self,
            old_vector: dict,
            incoming_vector: dict,
            persistent_vector: dict = None,
            alpha: float = 0.2,
            nonlinear_gain: float = 0.4,
            max_delta: float = 0.15,
            beta: float = 0.3,      # how much persistent baseline influences the target
        ):
        """
        Update an affect vector with a new observation, optionally influenced by
        a persistent (long-term) affect baseline AND maintain textual reasons.

        Behaviour on reasons
        --------------------
        - If trend (delta) > 0: keep existing reasons and add incoming reasons.
        - If trend (delta) < 0: progressively drop old reasons (fade out memory).
        - Always cap to `max_reasons`.
        """
        new_vector = {}
        trend = {}

        for emotion in old_vector.keys():
            old_entry = old_vector.get(emotion, 0.0)
            inc_entry = incoming_vector.get(emotion, {"value": 0.0, "reasons": []})

            old_val, old_reasons = self._extract_value_and_reasons(old_entry)
            inc_val, inc_reasons = self._extract_value_and_reasons(inc_entry)
            # persistent baseline value (no reasons needed here)
            if persistent_vector is not None:
                base_entry = persistent_vector.get(emotion, 0.0)
                base_val, _ = self._extract_value_and_reasons(base_entry)
                inc_val = (1.0 - beta) * inc_val + beta * base_val
            # Base EMA update
            ema_val = (1.0 - alpha) * old_val + alpha * inc_val
            # Non-linear boost (stronger update when inc ≠ old)
            diff = inc_val - old_val
            nonlinear_component = nonlinear_gain * (diff ** 3)
            raw_new = ema_val + nonlinear_component
            # Clipping to avoid large jumps
            delta = raw_new - old_val
            if delta > max_delta:
                delta = max_delta
            elif delta < -max_delta:
                delta = -max_delta
            final_val = old_val + delta
            # Keep values in [0,1]
            if final_val < 0.0:
                final_val = 0.0
            elif final_val > 1.0:
                final_val = 1.0
            # ----- Reason handling -----
            # Normalize magnitude in [0,1] for reason decay
            norm_mag = min(1.0, abs(delta) / max_delta) if max_delta > 0 else 0.0
            if delta >= 0:
                # Emotion increasing or stable: keep old reasons and add new ones
                merged = old_reasons + inc_reasons
                # Simple de-dup while preserving order
                seen = set()
                filtered = []
                for r in merged:
                    if r not in seen:
                        seen.add(r)
                        filtered.append(r)
                new_reasons = filtered[-self.max_reasons_per_emotion:]
            else:
                # Emotion decreasing: fade out reasons
                # Keep only a fraction of old reasons depending on how fast it goes down
                if old_reasons:
                    keep_frac = max(0.0, 1.0 - norm_mag)  # more negative delta => smaller keep_frac
                    keep_n = int(round(len(old_reasons) * keep_frac))
                    keep_n = max(0, min(len(old_reasons), keep_n))
                    new_reasons = old_reasons[-keep_n:]
                else:
                    new_reasons = []
                # Optionally, still consider new reasons a bit
                # but keep them weak so we don't "argue against" the decreasing trend
                if inc_reasons:
                    for r in inc_reasons:
                        if r not in new_reasons:
                            new_reasons.append(r)

                new_reasons = new_reasons[-self.max_reasons_per_emotion:]
            new_vector[emotion] = {
                "value": final_val,
                "reasons": new_reasons
            }
            trend[emotion] = delta

        return new_vector, trend