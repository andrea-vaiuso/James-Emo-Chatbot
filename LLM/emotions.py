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

        self.print_reason_sumup_model_warning = True

    def analyze_emotional_state(self, text: str, emotion_threshold=0.4) -> dict:
        if self.reason_sumup_model is None or self.reason_sumup_tokenizer is None:
            if self.print_reason_sumup_model_warning:
                print("⚠️ Warning: Reason summarization model/tokenizer not provided. Reasons will be empty.")
                self.print_reason_sumup_model_warning = False
        results = self.classifier(text)
        results = results[0]
        # rename key "label" to "emotion", key "score" to "value" and add "reasons" key with empty list
        formatted_results = [
            {"emotion": res["label"], "value": res["score"], "reasons": []}
            for res in results
        ]
        # order the list by score descending
        formatted_results.sort(key=lambda x: x["value"], reverse=True)
        # Add reasons summarization for emotions with values above threshold
        for res in formatted_results:
            emotion = res["emotion"]
            value = res["value"]
            if value > emotion_threshold and self.reason_sumup_model is not None and self.reason_sumup_tokenizer is not None:
                summary = self._summarize_reasons(text, emotion)
                res["reasons"].append(summary)
        # transform the list into a dict where keys are emotions
        formatted_results_dict = {res["emotion"]: {"value": res["value"], "reasons": res["reasons"]} for res in formatted_results}
        return formatted_results_dict
        
    def _summarize_reasons(self, text: str, emotion:str, max_new_tokens=50) -> str:
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are a language model tasked with summarizing reasons for feeling {emotion}.\n"
            f"Summarize the following reasons into a concise and complete phrase of maximum {max_new_tokens} tokens:\n"
            f"Do not write anything else than the summary.\n"
            f"Example format: 'I feel {emotion} because ...'\n"
            "Reasons:\n"
            f"{text}"
            "<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"Summary: "
        )
        inputs = self.reason_sumup_tokenizer(prompt, return_tensors="pt").to(self.reason_sumup_model.device)
        gen_ids = self.reason_sumup_model.generate(inputs.input_ids, 
            max_new_tokens=max_new_tokens,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            eos_token_id=self.reason_sumup_tokenizer.eos_token_id,
            pad_token_id=self.reason_sumup_tokenizer.eos_token_id,
            )
        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        summary = self.reason_sumup_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return summary
    
    @staticmethod
    def intensity_phrase(v: float) -> str:
        """Map value → human intensity phrase."""
        if v >= 0.9:
            return "an overwhelming amount of"
        if v >= 0.7:
            return "a lot of"
        if v >= 0.45:
            return "quite a bit of"
        if v >= 0.2:
            return "a bit of"
        if v > 0.0:
            return "a slight amount of"
        return None
