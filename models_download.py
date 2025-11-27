
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import os
set_seed(42)

MODEL_DIR = "./models/"

if __name__ == "__main__":
    # if models dir is not there, create it and download models
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        MODELSLIST = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen3-4B-Instruct-2507",
            "01-ai/Yi-1.5-9B-Chat",
            "microsoft/Phi-3-medium-128k-instruct"
        ]
        for model_name in MODELSLIST :
            # laoding model and tokenizer
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            save_dir = MODEL_DIR + f"{model_name.replace('/', '_')}"
            # saving model and tokenizer
            ref_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            # cleaning up to free memory
            del ref_model
            del tokenizer

    