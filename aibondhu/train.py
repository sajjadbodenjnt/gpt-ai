import argparse, random, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from .model import AIBondhuGPT
from .data import TextDataset

def evaluate(model, loader, device, max_batches):
    model.eval(); losses=[]
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            if i >= max_batches: break
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses) if losses else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config, encoding="utf-8"))
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device="cuda" if cfg["train"]["device"]=="auto" and torch.cuda.is_available() else ("cpu" if cfg["train"]["device"]=="auto" else cfg["train"]["device"])
    ds=TextDataset(cfg["data_path"], cfg["model"]["block_size"])
    if len(ds)<2: raise SystemExit("Not enough tokens for training; add more authorized text data.")
    n_val=max(1, int(len(ds)*0.1)); n_train=len(ds)-n_val
    train_ds,val_ds=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(cfg["seed"]))
    tr=DataLoader(train_ds,batch_size=cfg["train"]["batch_size"],shuffle=True,num_workers=cfg["train"]["num_workers"],drop_last=True)
    va=DataLoader(val_ds,batch_size=cfg["train"]["batch_size"],shuffle=False)
    model=AIBondhuGPT(cfg["model"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["learning_rate"],weight_decay=cfg["train"]["weight_decay"])
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    best=float("inf"); step=0; epoch=0
    while step<cfg["train"]["max_steps"]:
        epoch+=1
        for x,y in tqdm(tr,desc=f"epoch {epoch}"):
            model.train(); x,y=x.to(device),y.to(device)
            _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["grad_clip"]); opt.step()
            step+=1
            if step%cfg["train"]["eval_interval"]==0 or step==cfg["train"]["max_steps"]:
                val=evaluate(model,va,device,cfg["train"]["eval_batches"])
                print(f"step={step} train_loss={loss.item():.4f} val_loss={val:.4f}")
                state={"step":step,"model":model.state_dict(),"config":cfg["model"],"val_loss":val}
                torch.save(state,out/"latest.pt")
                if val<best:
                    best=val; torch.save(state,out/"best.pt")
            if step>=cfg["train"]["max_steps"]: break
    print(f"Training finished. best_val_loss={best:.4f}; device={device}")

if __name__=="__main__": main()
