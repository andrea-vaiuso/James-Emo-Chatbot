from datetime import datetime
import os
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, pipeline
from typing import Optional
from LLM.emotions import EmotionalStateManager
import json
import hashlib

def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_user_id(user_name):
    return hashlib.sha256(user_name.encode('utf-8')).hexdigest()

class MemoryManager:
    def __init__(self, memory_location, ai_username, user_name, 
                 llm_model: PreTrainedModel=None, tokenizer: PreTrainedTokenizer=None,
                 memory_template: dict=None,
                 user_age: Optional[str] = None,
                 user_gender: Optional[str] = None,
                 max_short_term_memory_entries=20):
        
        self.ai_username = ai_username
        self.user_name = user_name
        self.user_age = user_age
        self.user_gender = user_gender

        self.memory_chunks = [] # Contains the current session's text messages chunks
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.max_short_term_memory_entries = max_short_term_memory_entries

        if not os.path.exists(memory_location):
            os.makedirs(memory_location) 
        self.memory_path = f"{memory_location}/{ai_username}_memory.json"
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w", encoding="utf-8") as f:
                self.memory_database = memory_template
                self.memory_database["meta"]["user_id"] = generate_user_id(user_name)
                self.memory_database["meta"]["last_update"] = now()
                self.memory_database["meta"]["first_interaction"] = True
                # Mark as unknown so first prompt knows this is the first time
                self.memory_database["meta"]["last_interaction"] = "Unknown"
                self.memory_database["bot_profile"]["name"] = ai_username
                self.memory_database["user_profile"]["name"] = user_name
                if self.user_age:
                    self.memory_database["user_profile"]["age"] = str(self.user_age)
                if self.user_gender:
                    self.memory_database["user_profile"]["gender"] = str(self.user_gender)
                json.dump(self.memory_database, f)
        else:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.memory_database = json.load(f)
                self.memory_database["meta"]["first_interaction"] = False
                if self.user_age and self.memory_database["user_profile"].get("age") in (None, "", "Unknown"):
                    self.memory_database["user_profile"]["age"] = str(self.user_age)
                    self._dump_memory_database()
                if self.user_gender and self.memory_database["user_profile"].get("gender") in (None, "", "Unknown", "Not specified"):
                    self.memory_database["user_profile"]["gender"] = str(self.user_gender)
                    self._dump_memory_database()
        
        if "short_term_memory" not in self.memory_database:
            self.memory_database["short_term_memory"] = {
                "recent_summaries": [],
                "affect_state": {label: 0.0 for label in self.esm.labels}
            }
        if "long_term_memory" not in self.memory_database:
            self.memory_database["long_term_memory"] = {
                "facts": [],
                "persistent_affect_state": {
                    label: {"value": 0.0, "reasons": []}
                    for label in self.esm.labels
                }
            }
        
        self.esm = EmotionalStateManager(reason_sumup_model=self.llm_model,
                                         reason_sumup_tokenizer=self.tokenizer)

    def _dump_memory_database(self):
        self.memory_database["meta"]["last_update"] = now()
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_database, f, ensure_ascii=False, indent=4)
        print(f"Memory updated")

    def _generate_summary(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        print("Generating summary...")
        gen_ids = self.llm_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        summary = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return summary
    
    def get_current_emotional_state(self):
        return self.memory_database["current_affect_state"]
    
    def update_current_emotional_state(self, text: str):
        emo_state = self.esm.analyze_emotional_state(text)
        self.memory_database["current_affect_state"] = emo_state
    
    def _update_short_term_emotional_state(self, text: str):
        emo_state = self.esm.analyze_emotional_state(text)
        self.memory_database["short_term_memory"]["affect_state"] = self.esm.update_affect_state(
            self.memory_database["short_term_memory"]["affect_state"],
            emo_state,
            self.memory_database["long_term_memory"]["persistent_affect_state"],
            alpha=0.2,
            nonlinear_gain=0.4,
            max_delta=0.15,
            beta=0.3,
        )[0]

    
    def _update_long_term_emotional_state(self, text: str):
        emo_state = self.esm.analyze_emotional_state(text)
        self.memory_database["long_term_memory"]["persistent_affect_state"] = self.esm.update_affect_state(
            self.memory_database["long_term_memory"]["persistent_affect_state"],
            emo_state,
            None,
            alpha=0.05,
            nonlinear_gain=0.2,
            max_delta=0.05,
            beta=0.0,
        )[0]
        
    def _generate_short_term_memory_chunk(self, max_new_tokens=200, append_n_messages=3):
        if self.memory_chunks == []:
            return
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are human called {self.ai_username} tasked with compressing recent interactions into concise memory entries.\n"
            f"Summarize the recent interactions into a concise memory entry of maximum {max_new_tokens} tokens:\n"
            f"For each memory entry, write explicitly your name '{self.ai_username}' and the user's name '{self.user_name}'. Do not confuse the roles\n"
            f"Do not write anithing else than the memory entry.\n"
            "Session Messages:\n"
            f"{self.memory_chunks}"
            "<|eot_id|>"
            f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
            f"{now()} - Compressed Memory Entry: "
        )
        print("Generating memories...")
        compressed_memory = self._generate_summary(prompt, max_new_tokens=max_new_tokens)
        recent_messages = []
        if append_n_messages > 0 and len(self.memory_chunks) >= append_n_messages:
            recent_messages = self.memory_chunks[-append_n_messages:]
        self.memory_database["short_term_memory"]["recent_summaries"].append({
            "timestamp": now(),
            "summary": compressed_memory,
            "last_interactions": recent_messages
        })
        self._update_short_term_emotional_state(compressed_memory)
        # If too many short memory entries, create a long-term memory fact and reset short-term memory but the last one
        if len(self.memory_database["short_term_memory"]["recent_summaries"]) > self.max_short_term_memory_entries:
            self._generate_long_term_memory_fact()
            # Keep only the last short-term memory entry
            last_entry = self.memory_database["short_term_memory"]["recent_summaries"][-1]
            self.memory_database["short_term_memory"]["recent_summaries"] = [last_entry]
        self._dump_memory_database()
    
    def _generate_long_term_memory_fact(self, max_new_tokens=300):
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are an human called {self.ai_username} tasked with extracting significant facts from recent interactions.\n"
            f"Identify and summarize any significant facts or events from the recent interactions into a concise memory fact of maximum {max_new_tokens} tokens:\n"
            f"If there are no significant facts, respond with 'no significant interactions'\n"
            "Recent summaries:\n"
            f"{self.memory_database['short_term_memory']['recent_summaries']}"
            "<|eot_id|>"
            f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
            f"Significant Memory Fact: "
        )
        print("Generating long-term memory fact...")
        significant_fact = self._generate_summary(prompt, max_new_tokens=max_new_tokens)
        if not "no significant interactions" in significant_fact.lower():
            self.memory_database["long_term_memory"]["facts"].append({
                "timestamp": now(),
                "summary": significant_fact
            })
            self._update_long_term_emotional_state(significant_fact)
        self._dump_memory_database()

    def update_last_interaction(self): # To use in the end of an interaction
        self.memory_database["meta"]["last_interaction"] = now()
        self._dump_memory_database()
    
    def end_conversation(self, max_new_tokens=200, append_n_messages=3):
        self._generate_short_term_memory_chunk(
            max_new_tokens=max_new_tokens,
            append_n_messages=append_n_messages
        )
        self.update_last_interaction()


    def get_bot_informational_prompt(self) -> str:
        """
        Generate bot informational prompt based on bot profile.
        """
        bot_profile = self.memory_database["bot_profile"]
        prompt = (
            "Your Profile Information:\n"
            f"You name is {bot_profile.get('name', self.ai_username)}\n"
            f"You are {bot_profile.get('age', '25')} years old\n"
            f"Personality: {bot_profile.get('personality_traits', 'N/A')}\n"
            f"In the last days this happened to you: {bot_profile.get('recent_activities', 'nothing special')}\n"
        )
        return prompt

    def get_user_informational_prompt(self) -> str:
        """
        Generate user informational prompt based on user profile.
        """
        user_profile = self.memory_database["user_profile"]
        user_gender = user_profile.get("gender", "Not specified")
        if user_gender.lower().startswith("m"):
            pronoun = "he/him"
        elif user_gender.lower().startswith("f"):
            pronoun = "she/her"
        else:
            pronoun = "they/them"
        prompt = (
            "User Profile Information:\n"
            f"User is called {user_profile.get('name', 'User')}\n"
            f"Preferred pronouns: {pronoun}\n"
        )
        return prompt

    def get_memory_prompt(self) -> str:
        """
        Generate memory prompt including short-term and long-term memories.
        """
        short_term_memories = self.memory_database["short_term_memory"]["recent_summaries"]
        long_term_memories = self.memory_database["long_term_memory"]["facts"]

        short_term_prompt = "Short-Term Memories:\n"
        for entry in short_term_memories:
            short_term_prompt += f"- {entry['summary']}\n"

        long_term_prompt = "Long-Term Memories:\n"
        for entry in long_term_memories:
            long_term_prompt += f"- {entry['summary']}\n"

        return short_term_prompt + "\n" + long_term_prompt + "\n"

    def get_timing_information_prompt(self) -> str:
        """
        Generate timing information prompt based on last interaction.
        The generated prompt should inform the LLM about how much time has passed since the last interaction in days/hours/minutes.
        """
        meta = self.memory_database.get("meta", {})
        if meta.get("first_interaction", False):
            return "Timing Information: This is your very first conversation with the user; you have no prior interactions.\n"

        last_interaction_str = meta.get("last_interaction", None)
        if last_interaction_str in (None, "", "Unknown"):
            return ""
        last_interaction = datetime.strptime(last_interaction_str, "%Y-%m-%dT%H:%M:%SZ")
        now_dt = datetime.utcnow()
        delta = now_dt - last_interaction
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        time_parts = []
        if days > 0:
            time_parts.append(f"{days} days")
        if hours > 0:
            time_parts.append(f"{hours} hours")
        if minutes > 0:
            time_parts.append(f"{minutes} minutes")
        
        if not time_parts:
            time_parts.append("less than a minute")

        time_passed_str = ", ".join(time_parts)
        return f"Timing Information: The last interaction with the user was {time_passed_str} ago.\n"
    
    def get_emotional_prompt(self) -> str:
        """
        Generate a descriptive emotional context prompt using short-term and long-term emotional states.
        - Short-term: describes how intensely each emotion is felt "right now".
        - Long-term: describes stable emotional tendencies and the key reasons behind them.
        """

        short_term_emo = self.memory_database["short_term_memory"]["affect_state"]
        long_term_emo = self.memory_database["long_term_memory"]["persistent_affect_state"]

        # ------------------ UTILITIES ------------------

        def extract(entry):
            """Return (value, reasons_list) regardless of structure."""
            if isinstance(entry, dict):
                return float(entry.get("value", 0.0)), list(entry.get("reasons", []))
            return float(entry), []

        def choose_top(emo_dict, top_n=5, min_val=0.15):
            """Return list of (label, value, reasons) sorted by value."""
            vals = []
            for emo, entry in emo_dict.items():
                v, r = extract(entry)
                if v >= min_val:
                    vals.append((emo, v, r))
            vals.sort(key=lambda x: x[1], reverse=True)
            return vals[:top_n]

        def intensity_phrase(v):
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

        def clean_reasons(reasons, max_len=3):
            """Return a short, safe list of reasons."""
            if not reasons:
                return []
            # Trim and filter empty garbage
            r = [x.strip() for x in reasons if x and x.strip()]
            if not r:
                return []
            # Limit to the last ones, which are the freshest
            return r[-max_len:]

        # ------------------ SHORT TERM ------------------

        st = choose_top(short_term_emo, top_n=5, min_val=0.15)

        if not st:
            short_section = (
                "Short-term emotional state (right now):\n"
                "You are not clearly dominated by any strong emotion at this moment.\n"
            )
        else:
            lines = ["Short-term emotional state (right now):"]
            for emo, v, reasons in st:
                intensity = intensity_phrase(v)
                if not intensity:
                    continue

                base = f"- You are now feeling {intensity} '{emo}'."

                clean_r = clean_reasons(reasons, max_len=2)
                if clean_r:
                    rs = "; ".join(clean_r)
                    lines.append(f"{base} It seems connected to: {rs}.")
                else:
                    lines.append(base)

            short_section = "\n".join(lines) + "\n"

        # ------------------ LONG TERM ------------------

        lt = choose_top(long_term_emo, top_n=5, min_val=0.20)

        if not lt:
            long_section = (
                "Long-term emotional tendencies:\n"
                "There are no stable long-term emotional patterns identified yet.\n"
            )
        else:
            lines = ["Long-term emotional tendencies:"]
            for emo, v, reasons in lt:
                base = f"- Over time, you show a tendency toward '{emo}'."

                clean_r = clean_reasons(reasons, max_len=3)
                if clean_r:
                    rs = "; ".join(clean_r)
                    lines.append(f"{base} This long-standing pattern seems influenced by: {rs}.")
                else:
                    lines.append(
                        f"{base} This is inferred from stable patterns in past interactions."
                    )

            long_section = "\n".join(lines) + "\n"

        # ------------------ FINAL ASSEMBLY ------------------

        intro = (
            "The following information describes the user's emotional state.\n"
            "Use it to express what you feel.\n"
            "Interpret emotions as an human.\n"
        )

        return intro + "\n" + short_section + "\n" + long_section


            
            
