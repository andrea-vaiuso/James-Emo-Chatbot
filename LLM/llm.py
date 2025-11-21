
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
                 user_name="User",
                 user_age=None,
                 user_gender=None):
        self.ai_username = ai_username
        self.user_name = user_name
        self.user_age = user_age
        self.user_gender = user_gender
        self.personality_prompt_file = personality_prompt_file
        with open(personality_prompt_file, "r", encoding="utf-8") as f:
            self.ai_personality_prompt = f.read().strip()
        self.model, self.tokenizer = self.init_model(model_name)
        self.memory_manager = MemoryManager(memory_location, 
                                            ai_username=ai_username, 
                                            user_name=user_name, 
                                            llm_model=self.model, 
                                            tokenizer=self.tokenizer,
                                            memory_template=self.load_memory_template(),
                                            user_age=user_age,
                                            user_gender=user_gender)
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

    def get_system_prompt(self, verbose=False) -> str:
        context_block = self.memory_manager.get_memories_for_prompt()
        prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{self.ai_personality_prompt}\n"
            f"{context_block}\n"
            "<|eot_id|>"
        )
        if verbose:
            print( "=== CONVERSATION PROMPT ===")
            print(prompt)
            print( "===========================")
        return prompt
    
    def intensity_phrase(self,v):
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
    
    def generate_response(self, conversation_history, emotion_threshold=0.4, n_top_emotions=2, include_emotional_context: bool = False) -> str:
        if include_emotional_context:
            current_emotional_state = self.memory_manager.get_current_emotional_state()
            # Get the two higher emotions other than neutral and build emotional context
            sorted_emotions = sorted(
                current_emotional_state.items(),
                key=lambda item: item[1]["value"],
                reverse=True
            )
            top_emotions = [f"I feel {self.intensity_phrase(data['value'])} {emotion}" for emotion, data in sorted_emotions if data["value"] > emotion_threshold][:n_top_emotions]
            emotional_context = ". ".join(top_emotions) if top_emotions else None
            if emotional_context is not None:
                emotional_prompt = (
                    "<|start_header_id|>system<|end_header_id|>"
                    f"These are the emotions I am currently feeling: {emotional_context}.\n"
                    "<|eot_id|>"
                )
                conversation_history += emotional_prompt
                print(f"Emotional Context: {emotional_context}")
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
        # self.memory_manager.update_current_emotional_state(response)
        return response
    
    def end_conversation(self):
        self.memory_manager.end_conversation()
    
    @property
    def memory_chunks(self):
        return self.memory_manager.memory_chunks
    
    
