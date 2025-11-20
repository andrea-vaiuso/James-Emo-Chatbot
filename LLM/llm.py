
from datetime import datetime
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, PreTrainedModel
import torch
from utilities import get_current_location
from LLM.memory import MemoryManager
import os
import json

class LLMModel:

    def __init__(self, ai_username, personality_prompt_file, profile_picture_path=None,
                 model_name="meta-llama/Llama-3.1-8B-Instruct",
                 memory_location="Memory",
                 user_name="User"):
        self.ai_username = ai_username
        self.user_name = user_name
        self.personality_prompt_file = personality_prompt_file
        with open(personality_prompt_file, "r", encoding="utf-8") as f:
            self.ai_personality_prompt = f.read().strip()
        self.model, self.tokenizer = self.init_model(model_name)
        self.memory_manager = MemoryManager(memory_location, 
                                            ai_username=ai_username, 
                                            user_name=user_name, 
                                            llm_model=self.model, 
                                            tokenizer=self.tokenizer,
                                            memory_template=self.load_memory_template())
        self.profile_picture_path = profile_picture_path

    def load_memory_template(self):
        template_path = os.path.join(os.path.dirname(self.personality_prompt_file), "memory_template.json")
        with open(template_path, "r", encoding="utf-8") as f:
            memory_template = json.load(f)
        return memory_template

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
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
    
    from datetime import datetime

    def init_conversation(self, user_name) -> str:
        memory_db = self.memory_manager.memory_database
        meta = memory_db.get("meta", {})

        # --- Informational sub-prompts from MemoryManager ---
        bot_info_prompt = self.memory_manager.get_bot_informational_prompt()
        user_info_prompt = self.memory_manager.get_user_informational_prompt()
        memory_prompt = self.memory_manager.get_memory_prompt()
        timing_prompt = self.memory_manager.get_timing_information_prompt()
        emotional_prompt = self.memory_manager.get_emotional_prompt()

        # --- First interaction vs previous interactions ---
        last_interaction = meta.get("last_interaction", None)
        is_first_interaction = last_interaction in (None, "Unknown")

        if is_first_interaction:
            relationship_instruction = (
                f"This is your first conversation with {user_name}. "
                "Warmly welcome them, ask a few simple questions to get to know them, "
                "and build rapport in a natural way.\n"
            )
        else:
            relationship_instruction = (
                f"You have already talked with {user_name} before. "
                "Use your memories and the information below to maintain continuity, "
                "refer back to past topics when helpful, and avoid asking the same basic questions again.\n"
            )

        # --- Context block ---
        context_block = (
            "Context Information. Always use this info to ground your answers where relevant:\n"
            f"- Today is {datetime.now().strftime('%B %d, %Y')}.\n"
            f"- You are writing text messages in WhatsApp as '{self.ai_username}'. "
            "Use WhatsApp-like style (short messages, informal but clear language).\n"
            "- You are a pen pal: you do not know the exact physical location of the user "
            "and you cannot organize in-person meetings because distance, so your relationship is limited to written communication.\n"
            f"- The person you are talking to is called {user_name}. Remember and use this name naturally.\n"
            f"{emotional_prompt}"
            f"{relationship_instruction}"
            f"{timing_prompt}"
            f"{bot_info_prompt}"
            f"{user_info_prompt}"
            f"{memory_prompt}"
        )

        # --- Final system prompt ---
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{self.ai_personality_prompt}\n"
            f"{context_block}\n"
            "<|eot_id|>"
        )

        return prompt

    
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
    
    def end_conversation(self):
        self.memory_manager.end_conversation()
    
    @property
    def memory_chunks(self):
        return self.memory_manager.memory_chunks
    
    
