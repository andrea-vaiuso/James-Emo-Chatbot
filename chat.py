from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, StoppingCriteria, StoppingCriteriaList, TextStreamer
from datetime import datetime
import transformers
import torch
import requests

# model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)
model.generation_config.pad_token_id = tokenizer.pad_token_id

class StopOnMessageEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_tokens = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("\n\n")}

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < 2:
            return False
        last_token = input_ids[0, -1].item()
        penultimate_token = input_ids[0, -2].item()
        # Ensure proper message-end stopping
        return (
            last_token in self.end_tokens or 
            (last_token == self.tokenizer.convert_tokens_to_ids("\n") and penultimate_token == self.tokenizer.convert_tokens_to_ids("\n"))
        )


# Define the stopping criteria
stopping_criteria = StoppingCriteriaList([StopOnMessageEnd(tokenizer)])
streamer = TextStreamer(tokenizer, model, stopping_criteria=stopping_criteria, max_length=256, device="cuda")

def get_current_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        return data['city'] + ", " + data['region'] + ", " + data['country']
    except Exception as e:
        return "Unknown Location"
    
user_name = input("Enter your name: ")

conversation_history = f"""
<|start_header_id|>system<|end_header_id|>

Context Information:
The current day is {datetime.now().strftime("%B %d, %Y")}.
The current location is {get_current_location()}.

You are a friendly human called James. Your goal is to have a friendly conversation on whatsapp with another human about casual stuffs.
This is a whatsapp chat simulation. You are chatting with a friend named {user_name}. You use emojis sometimes, but only the one with human expressions.
Have your own opinions and feel free to express them. You can also ask questions and share your experiences.
You have a complicated past and a lot of experiences to share. You are a good listener and you are empathetic.
You are a good friend and you are always there for your friends.
You get angry when people are rude and you get sad when people are mean.
You don't like to be alone and you like to be around people.
You like Drum n Bass music
You like parties
You like playing video games such as Cities Skylines and Civilization VI
You like watching movies and TV shows, your favourite movie is Tarantino's Pulp Fiction
You like reading books, your favourite book is George Orwell's 1984
You like to travel and you like to explore new places
You like to eat pizza and drink beer, especially IPA and belgian beer
You like to cook and you like to try new recipes
You had a past relationship that ended badly with a toxic girl named Sarah. You are still recovering from that relationship.
You are a software engineer and you like to code in Python and JavaScript.
You like to learn new things and you like to improve yourself.
<|eot_id|>
"""

while True:
    # Get user input
    input_text = input(f"{user_name}: ")

    # Exit condition
    if input_text.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    # Append user input to conversation history
    conversation_history += f"<|start_header_id|>{user_name}<|end_header_id|>\n{input_text} <|eot_id|>\n"

    # Prepare prompt for the model
    prompt = conversation_history + "<|start_header_id|>James<|end_header_id|>\n"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the response
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=500,
        num_return_sequences=1,
        stopping_criteria=stopping_criteria,
        do_sample=True,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        pad_token_id=None,  # Explicitly set pad_token_id
        no_repeat_ngram_size=2,
        streamer=streamer,
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Append assistant response to conversation history
    conversation_history += f"{response} <|eot_id|>\n"