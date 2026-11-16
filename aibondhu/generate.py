import argparse, torch
from .model import AIBondhuGPT
from .tokenizer import ByteTokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",required=True)
    p.add_argument("--tokens",type=int,default=120); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=AIBondhuGPT(ck["config"]); model.load_state_dict(ck["model"]); model.eval()
    tok=ByteTokenizer(); x=torch.tensor([tok.encode(a.prompt)],dtype=torch.long)
    print(tok.decode(model.generate(x,max_new_tokens=a.tokens)[0].tolist()))
    print("\n[Experimental output: not medical advice. Consult a qualified clinician.]")

if __name__=="__main__": main()
