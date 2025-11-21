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
        
        self.esm = EmotionalStateManager(reason_sumup_model=self.llm_model,
                                         reason_sumup_tokenizer=self.tokenizer)

    def _dump_memory_database(self):
        self.memory_database["meta"]["last_update"] = now()
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_database, f, ensure_ascii=False, indent=4)
        print(f"Memory updated")

    def _model_generate(self, prompt, max_new_tokens):
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
    
    def get_current_emotional_state(self) -> list:
        return self.memory_database["current_affect_state"]
    
    def update_emotional_value(self, old: float, new: float, alpha: float = 0.6) -> float:
        """Update emotional value using exponential moving average."""
        # TODO: To update considering long-term emotional state
        updated_value = alpha * new + (1 - alpha) * old
        return updated_value
    
    # TODO: Insert a function to be executed in a thread to decay emotional state over time and, if the anticipation is high, eventually turn on a flag to activate proactive new messages.
    
    def update_current_emotional_state(self, text: str):
        emo_state = self.esm.analyze_emotional_state(text)
        # For each emotion in emo_state, update the memory database current_affect_state by updating value and appending reasons
        for emo in emo_state.keys():
            new_emotional_value = self.update_emotional_value(
                old = self.memory_database["current_affect_state"].get(emo, {"value": 0.0, "reasons": []})["value"],
                new = emo_state[emo]["value"]
            )
            if emo not in self.memory_database["current_affect_state"].keys():
                self.memory_database["current_affect_state"][emo] = {"value": 0.0, "reasons": []}
            self.memory_database["current_affect_state"][emo]["value"] = new_emotional_value
            if emo_state[emo]["reasons"]:
                    self.memory_database["current_affect_state"][emo]["reasons"] = emo_state[emo]["reasons"]
                
        self.memory_database["current_affect_state"] = emo_state
    
    def get_current_emotional_state(self) -> dict:
        return self.memory_database["current_affect_state"]
        
    def generate_short_term_memory_chunk(self, max_new_tokens=200, append_n_messages=3):
        if self.memory_chunks == []:
            return
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are human called {self.ai_username} tasked with compressing recent interactions into concise memory entries like you are writing a diary entry.\n"
            f"Summarize the recent interactions into a concise memory entry of maximum {max_new_tokens} tokens:\n"
            f"For each memory entry, write explicitly your name '{self.ai_username}' and the user's name '{self.user_name}'. Do not confuse the roles\n"
            f"Do not write anithing else than the memory entry.\n"
            f"Separate multiple memories using '.'\n"
            f"Write memories in plain text.\n"
            "Session Messages:\n"
            f"{self.memory_chunks}"
            "<|eot_id|>"
            f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
            f"Compressed Memory Entry: "
        )
        print("Generating memories...")
        # Generate compressed memory entry
        compressed_memory = self._model_generate(prompt, max_new_tokens=max_new_tokens)
        # Append recent messages to the memory entry
        recent_messages = []
        if append_n_messages > 0 and len(self.memory_chunks) >= append_n_messages:
            recent_messages = self.memory_chunks[-append_n_messages:]
        # Add associated emotions to the memory entry
        associated_emotions = self.get_current_emotional_state()
        # Store in memory database
        self.memory_database["short_term_memory"]["memories"].append({
            "timestamp": now(),
            "summary": compressed_memory,
            "last_interactions": recent_messages,
            "associated_emotions": associated_emotions
        })
        
        # If too many short memory entries, create a long-term memory fact and reset short-term memory but the last one
        if len(self.memory_database["short_term_memory"]["memories"]) > self.max_short_term_memory_entries:
            self._generate_long_term_memory_fact()
            # Keep only the last short-term memory entry
            last_entry = self.memory_database["short_term_memory"]["memories"][-1]
            self.memory_database["short_term_memory"]["memories"] = [last_entry]
        self._dump_memory_database()
    
    def _generate_long_term_memory_fact(self, max_new_tokens=300, emotion_new_tokens=50):
        summaries_of_memories = "\n".join(
            [mem["summary"] for mem in self.memory_database["short_term_memory"]["memories"]]
        )
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are an human called {self.ai_username} tasked with extracting significant facts from recent interactions like you are writing a diary entry.\n"
            f"Identify and summarize any significant facts or events from the recent interactions into a concise memory fact of maximum {max_new_tokens} tokens:\n"
            f"Try to memorize as much as possible of yourself and the user as short facts.\n"
            f"Important facts are those that define personality, preferences, habits, routines, important events and changes in your life or the user's life, recurring themes, logistic informations.\n"
            f"For each fact, write explicitly your name '{self.ai_username}' and the user's name '{self.user_name}'. Do not confuse the roles\n"
            f"Example format: 'I, {self.ai_username}, like hiking on weekends.' or '{self.user_name} enjoys cooking Italian food.'\n"
            f"Separate multiple facts using '.'\n"
            f"Write facts in plain text.\n"
            "Recent summaries:\n"
            f"{summaries_of_memories}"
            "<|eot_id|>"
            f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
            f"Significant Memory Facts: "
        )
        print("Generating long-term memory fact...")
        significant_fact = self._model_generate(prompt, max_new_tokens=max_new_tokens)
        self.memory_database["long_term_memory"]["memories"].append({
            "timestamp": now(),
            "summary": significant_fact
        })
        list_of_memories_emotions = [
            mem["associated_emotions"] for mem in self.memory_database["short_term_memory"]["memories"]
        ]
        plain_text_emotions = ""
        for emo_dict in list_of_memories_emotions:
            for emo, data in emo_dict.items():
                plain_text_emotions += f"- I feel {EmotionalStateManager.intensity_phrase(data['value'])} {emo} because: {[r+"\n" for r in data['reasons']]}\n"
        emotion_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"You are an human called {self.ai_username} tasked with summarizing your most important emotions.\n"
            f"Summarize your most important persistent emotions based on your recent interactions into a concise emotional state description:\n"
            f"You are going to be provided with a set of emotions and their associated values and reasons.\n"
            f"Select the most important emotions that define your long-term emotional tendencies for the reasons that for you are more important.\n"
            f"You can write multiple emotions in a plain paragraph, like you are writing a diary entry.\n"
            f"Do not write more than {emotion_new_tokens} tokens in total, but finish your sentences.\n"
            "Recent emotional states:\n"
            f"{plain_text_emotions}"
            "<|eot_id|>"
            f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
            f"Persistent Emotional State Description: "
        )
        print("Generating long-term persistent emotional state...")
        self.memory_database["long_term_memory"]["persistent_emotions"].append(
            self._model_generate(emotion_prompt, max_new_tokens=emotion_new_tokens)
        )
        self._dump_memory_database()

    def _update_last_interaction(self): # To use in the end of an interaction
        self.memory_database["meta"]["last_interaction"] = now()
        self._dump_memory_database()
    
    def end_conversation(self, max_new_tokens=200, append_n_messages=3):
        self.generate_short_term_memory_chunk(
            max_new_tokens=max_new_tokens,
            append_n_messages=append_n_messages
        )
        self._update_last_interaction()

    def get_memories_for_prompt(self) -> str:
        bot_profile = self.memory_database["bot_profile"]
        user_profile = self.memory_database["user_profile"]
        user_gender = user_profile.get("gender", "Not specified")
        # f"In the last days this happened to you: {bot_profile.get('recent_activities', 'nothing special')}\n"
        if user_gender.lower().startswith("m"):
            pronoun = "he/him"
        elif user_gender.lower().startswith("f"):
            pronoun = "she/her"
        else:
            pronoun = "they/them"
        
        short_term_memories = self.memory_database["short_term_memory"]["memories"]
        long_term_memories = self.memory_database["long_term_memory"]["memories"]
        
        shrt_term_memories_paragraph = "\n".join(
            [f"- {entry['summary']}" for entry in short_term_memories]
        )
        long_term_memories_paragraph = "\n".join(
            [f"- {entry['summary']}" for entry in long_term_memories]
        )

        current_emotional_state = self.get_current_emotional_state()
        # get the first 3 higher emotions and build a paragraph (they are already sorted)
        current_emotional_state = dict(list(current_emotional_state.items())[:3])
        current_emotional_state_paragraph = "\n".join(
            [f"- I feel {EmotionalStateManager.intensity_phrase(data['value'])} {emotion}" + (f" because: {[r for r in data['reasons']]}" if data['reasons'] else "") for emotion, data in current_emotional_state.items() if data["value"] > 0.0]
        )

        persistent_emotions = self.memory_database["long_term_memory"].get("persistent_emotions", [])
        persistent_emotions_paragraph = ""
        if persistent_emotions:
            persistent_emotions_paragraph = "\n".join(
                [f"- {entry}" for entry in persistent_emotions]
            )
        persistent_emotions_section = (
            f"[Persistent Emotional Tendencies]:\n"
            f"These are your persistent emotional tendencies. They are part of your past conversations with this person, and you hardly ever change.\n"
            f"{persistent_emotions_paragraph}\n"
        ) if persistent_emotions_paragraph else ""

        prompt = (
            "[Your Profile Information]:\n"
            f"You name is {bot_profile.get('name', self.ai_username)}\n"
            f"You are {bot_profile.get('age', '25')} years old\n"
            f"Personality: {bot_profile.get('personality_traits', 'N/A')}\n"
            "Use WhatsApp-like style (short messages, informal but clear language).\n"
            f"\n"
            f"[{user_profile.get('name', 'User')} Profile Information]:\n"
            f"The person you are talking with is called {user_profile.get('name', 'User')}\n"
            f"{user_profile.get('name', 'User')} prefers these pronouns: {pronoun}\n"
            f"{user_profile.get('name', 'User')} is {user_profile.get('age', 'unknown age')} years old\n"
            f"\n"
            f"{self.get_timing_information_prompt()}"
            f"\n"
            f"[Memories:]\n"
            f"These are your memories. Always use them to ground your answers where relevant:\n"
            f"{long_term_memories_paragraph}\n"
            f"{shrt_term_memories_paragraph}\n"
            f"\n"
            f"[Current Emotional State]:\n"
            f"These are your current emotional states. Always use them to express your feelings where relevant.\n"
            f"{current_emotional_state_paragraph}\n"
            f"{persistent_emotions_section}"
        )
        return prompt

    def get_timing_information_prompt(self) -> str:
        """
        Generate timing information prompt based on last interaction.
        The generated prompt should inform the LLM about how much time has passed since the last interaction in days/hours/minutes.
        """
        meta = self.memory_database.get("meta", {})
        if meta.get("first_interaction", False):
            return "[Timing Information]:\n This is your very first conversation with the user; you have no prior interactions.\n"

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
        return f"[Timing Information]:\n The last interaction with the user was {time_passed_str} ago.\n"
    