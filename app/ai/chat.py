from fastapi import FastAPI
import os
import torch
os.environ['HF_HOME'] = '/app/ai/cache'

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str):
    """Load and return the model + tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        use_safetensors=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def generate_response(prompt: str, pre_prompt:str, model, tokenizer, max_new_tokens: int = 512) -> str:
    """Generate an answer for a given prompt."""
    messages = [
        {"role": "system", "content": pre_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# app = FastAPI()
model, tokenizer = load_model("Qwen/Qwen2.5-3B-Instruct")

# @app.get("/")
# def read_root():
prompt = "Écris un exemple de code PHP 8 avec un enum:string."
pre_prompt = "Tu es un assistant critique."
output = generate_response(prompt, pre_prompt, model, tokenizer)
# return {"answer": output}
print(output)
