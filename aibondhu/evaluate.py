import argparse, math, yaml, torch
from torch.utils.data import DataLoader
from .model import AIBondhuGPT
from .data import TextDataset

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--data",required=True)
    p.add_argument("--batch-size",type=int,default=4); a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=AIBondhuGPT(ckpt["config"]); model.load_state_dict(ckpt["model"]); model.eval()
    ds=TextDataset(a.data,ckpt["config"]["block_size"])
    if not len(ds): raise SystemExit("Evaluation dataset is empty or too small.")
    loader=DataLoader(ds,batch_size=a.batch_size); losses=[]
    with torch.no_grad():
        for x,y in loader:
            _,loss=model(x,y); losses.append(loss.item())
    mean=sum(losses)/len(losses)
    print({"tokens_windows":len(ds),"mean_loss":mean,"perplexity":math.exp(min(mean,20))})
    print("This is language-model fit only; it is not clinical validation.")

if __name__=="__main__": main()
