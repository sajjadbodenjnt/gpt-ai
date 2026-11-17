"""Train AIBondhu's own small GPT from random initialization on authorized PMC OA text."""
import argparse, json, math, random, time
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer
from aibondhu.scratch_gpt import ScratchGPT

def load_docs(path):
    docs=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj=json.loads(line); text=" ".join(str(obj.get("text","")).split())
                if len(text)>=300: docs.append(text)
            except (json.JSONDecodeError, AttributeError):
                continue
    # deterministic exact deduplication
    return list(dict.fromkeys(docs))

def encode_docs(tok, docs, eos):
    return [tok.encode(d).ids + [eos] for d in docs]

def batch_from(sequences, batch, block, device):
    xs=[]; ys=[]
    for _ in range(batch):
        seq=random.choice(sequences)
        if len(seq) < block+2:
            seq=(seq * ((block+2)//max(1,len(seq))+1))
        start=random.randint(0, len(seq)-block-2)
        xs.append(seq[start:start+block]); ys.append(seq[start+1:start+block+1])
    return torch.tensor(xs,dtype=torch.long,device=device), torch.tensor(ys,dtype=torch.long,device=device)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/processed.jsonl")
    p.add_argument("--output",default="outputs/aibondhu-scratch")
    p.add_argument("--minutes",type=int,default=300)
    p.add_argument("--max-steps",type=int,default=1000000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(max(1,min(2,torch.get_num_threads())))
    device=torch.device("cpu")
    docs=load_docs(a.input)
    if len(docs)<100:
        raise SystemExit(f"Need at least 100 distinct documents; found {len(docs)}")
    random.shuffle(docs)
    cut=max(1,int(len(docs)*0.95)); train_docs, val_docs=docs[:cut], docs[cut:]
    if not val_docs: val_docs=train_docs[-1:]; train_docs=train_docs[:-1]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    # This tokenizer is trained only on this corpus; no vendor model/tokenizer/weights.
    tokenizer=ByteLevelBPETokenizer()
    tok_path=out/"tokenizer"
    tok_path.mkdir(parents=True, exist_ok=True)
    tokenizer.train_from_iterator(train_docs, vocab_size=8000, min_frequency=2,
                                  special_tokens=["<|endoftext|>"])
    tokenizer.save_model(str(out/"tokenizer"))
    # ByteLevelBPETokenizer saves vocab/merges; reload the corpus-trained tokenizer.
    from tokenizers import ByteLevelBPETokenizer as BPE
    tok=BPE(str(tok_path/"vocab.json"),str(tok_path/"merges.txt"))
    eos=tok.token_to_id("<|endoftext|>")
    if eos is None: raise RuntimeError("EOS token missing")
    train=encode_docs(tok,train_docs,eos); val=encode_docs(tok,val_docs,eos)
    vocab=tok.get_vocab_size()
    model=ScratchGPT(vocab).to(device)
    params=sum(x.numel() for x in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1)
    scaler=None
    batch=8; block=256; eval_every=250; best=float("inf")
    deadline=time.monotonic()+max(1,a.minutes)*60
    step=0; started=time.monotonic()
    print(f"FROM-SCRATCH MODEL | params={params:,} | vocab={vocab} | docs={len(docs)} train={len(train)} val={len(val)} | CPU | target_minutes={a.minutes}",flush=True)
    while step<a.max_steps and time.monotonic()<deadline:
        model.train()
        x,y=batch_from(train,batch,block,device)
        _,loss=model(x,y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1
        if step%eval_every==0 or step==1:
            model.eval(); vals=[]
            with torch.no_grad():
                for _ in range(20):
                    vx,vy=batch_from(val,min(batch,4),block,device)
                    _,vl=model(vx,vy); vals.append(vl.item())
            vloss=sum(vals)/len(vals)
            elapsed=(time.monotonic()-started)/60
            print(f"step={step} train_loss={loss.item():.4f} val_loss={vloss:.4f} elapsed_min={elapsed:.1f}",flush=True)
            state={"step":step,"model":model.state_dict(),"vocab_size":vocab,"width":256,"layers":6,"heads":8,"block_size":block,"val_loss":vloss,"training":"random_initialization_no_pretrained_weights"}
            torch.save(state,out/"latest.pt")
            if vloss<best:
                best=vloss; torch.save(state,out/"best.pt")
        if step%1000==0:
            print(f"checkpoint heartbeat step={step}",flush=True)
    elapsed=(time.monotonic()-started)/60
    print(f"FINISHED steps={step} elapsed_minutes={elapsed:.1f} best_val_loss={best:.4f} params={params:,}",flush=True)
    (out/"training_summary.json").write_text(json.dumps({"steps":step,"elapsed_minutes":elapsed,"best_val_loss":best,"parameters":params,"documents":len(docs),"train_documents":len(train_docs),"validation_documents":len(val_docs),"pretrained_weights_used":False,"clinical_validation":False},indent=2),encoding="utf-8")

if __name__=="__main__": main()
