
from datetime import datetime
import os
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, PreTrainedModel
import torch
from utilities import get_current_location
import json
from PIL import Image

class MemoryManager:
    def __init__(self, memory_location, ai_username, user_name, 
                 llm_model: PreTrainedModel=None, tokenizer: AutoTokenizer=None):
        self.ai_username = ai_username
        self.user_name = user_name
        self.memory_database = {
            "memories": [
                {"summary": "", 
                 "timestamp": "", 
                 "emotions": [], 
                 "details_about_user": "", 
                 "details_about_me": "", 
                 "long_term_memory": []}
            ]
        }
        self.memory_chunks = []
        self.last_summary = ""

        self.llm_model = llm_model
        self.tokenizer = tokenizer

        if not os.path.exists(memory_location):
            os.makedirs(memory_location) 
        self.memory_path = f"{memory_location}/{ai_username}_memory.json"
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_database, f)

        with open(self.memory_path, "r", encoding="utf-8") as f:
            self.memory_database = json.load(f)
            memories_array = self.memory_database.get("memories", [])
            if memories_array:
                self.last_summary = memories_array[-1].get("summary", "")

    def get_conversation_timing_info(self):
        memories_array = self.memory_database.get("memories", [])
        if memories_array and memories_array[-1]["timestamp"]:
            last_timestamp_str = memories_array[-1]["timestamp"]
            last_timestamp = datetime.strptime(last_timestamp_str, "%Y-%m-%d %H:%M:%S")
            days_before_last_conversation = (datetime.now() - last_timestamp).days
            return last_timestamp_str, days_before_last_conversation
        else:
            return None, None
        
    def generate_compressed_memory(self, max_new_tokens=200, append_n_messages=3, write_to_file=True):
        """
        Generate a compressed summary of the given memory text. Append the last n messages if specified.
        Args:
            max_new_tokens (int): Maximum number of tokens to generate for the summary.
            append_n_messages (int): Number of recent messages to append to the summary.
        Returns:
            str: The compressed memory summary.
        """
        prompt = (
            f"You are '{self.ai_username}' and it's time to summarize your memories of conversations between you and '{self.user_name}'."
            f"The conversation is between you, a human called '{self.ai_username}', and another human user called '{self.user_name}'."
            f"Never confuse the identities of the two parties. When you write the memory summary, always refer to you and the user by name."
            f"Do not write more than {max_new_tokens} words."
            f"Do not write in your response anything else than the summary. No follow up explanations or comments."
            f"Mantain the most important details and events that happened to you and your user during this and the past conversations, since this summary will replace them."
            f"If there are none previous memories, just summarize the current conversation."
            f"\n\n Conversation you had with him in previous sessions:{self.last_summary if self.last_summary else ' No previous memories.'}"
            f":\n\n Conversation you had with him in this session:{''.join(self.memory_chunks)}\n\nSummary:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        print("Generating memories...")
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
        compressed_memory = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if append_n_messages > 0 and len(self.memory_chunks) >= append_n_messages:
            recent_messages = self.memory_chunks[-append_n_messages:]
            compressed_memory += "\n\nRecent Interactions:\n" + ''.join(recent_messages)
        if write_to_file:
            print("Saving memories...")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_memory_entry = {
                "summary": [compressed_memory],
                "timestamp": timestamp,
                "emotions": [], # Placeholder for future emotion analysis
                "details_about_user": "", # Placeholder for future user details
                "details_about_me": "", # Placeholder for future details about me
                "long_term_memory": [] # Placeholder for future long term memory entries
            }
            self.memory_database["memories"].append(new_memory_entry)
            self.memory_chunks = [compressed_memory]
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.memory_database, f, ensure_ascii=False, indent=4)
        print("Memory generation complete", f"and saved to {self.memory_path}." if write_to_file else ".")
        return compressed_memory

class LLMModel:

    def __init__(self, ai_username, personality_prompt_file, profile_picture_path=None,
                 model_name="meta-llama/Llama-3.1-8B-Instruct",
                 memory_location="Memory",
                 user_name="User"):
        self.ai_username = ai_username
        self.user_name = user_name
        with open(personality_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        self.model, self.tokenizer = self.init_model(model_name)
        self.memory_manager = MemoryManager(memory_location, 
                                            ai_username=ai_username, 
                                            user_name=user_name, 
                                            llm_model=self.model, 
                                            tokenizer=self.tokenizer)
        self.profile_picture_path = profile_picture_path

    def init_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load system prompt from config file

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    
    def init_conversation(self, user_name):
        last_timestamp_str, days_before_last_conversation = self.memory_manager.get_conversation_timing_info()
        if last_timestamp_str:
            timing_info = f"The last conversation you had with {user_name} was in date {last_timestamp_str} ({days_before_last_conversation} days ago)."
        else:
            timing_info = f"You have never had a conversation with {user_name} before. Know that you are chatting them for the first time."
        return f"""
                <|start_header_id|>system<|end_header_id|>
                {self.system_prompt}
                Context Information. Alwyays use this info to ground your answers where relevant. They should aways be true:
                The current day is {datetime.now().strftime("%B %d, %Y")}.
                The current location is {get_current_location()}.
                The person you are talking to is called {user_name}.
                Here, you are writing text messages in WhatsApp as {self.ai_username}. Uses whatsapp style and conventions.
                {timing_info}
                The following is a summary of your memories and past experiences with the current user:
                {''.join(self.memory_manager.last_summary) if self.memory_manager.last_summary else "You have no memories with this user yet."}
                <|eot_id|>
                """
    
    def generate_response(self, conversation_history):
        prompt = conversation_history + f"<|start_header_id|>{self.ai_username}<|end_header_id|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response
    
    @property
    def memory_chunks(self):
        return self.memory_manager.memory_chunks
    
    
