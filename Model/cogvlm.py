import argparse
import cv2
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.TOKENIZER_PATH)
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.MODEL_PATH,
        torch_dtype = torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    )

    return tokenizer, model

def prepare_inputs(inputs):
    

