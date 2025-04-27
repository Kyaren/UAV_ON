import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

def load_Model_and_Tokenizer(args):
    tokenizer=LlamaTokenizer.from_pretrained(args.local_tokenizer)
    torch_type= torch.bfloat16 if args.bf16 else torch.float16
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model= AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True if args.quant else False,
        trust_remote_code=True
    ).to(device).eval()
    
    return model, tokenizer, device, torch_type

def preprocess(image):
    


def generate_Response(model, tokenizer, device, torch_type, images):
    if images is None:
        pass

