import torch
from aibondhu.model import AIBondhuGPT
from aibondhu.tokenizer import ByteTokenizer

def test_byte_roundtrip():
    t=ByteTokenizer()
    s="বাংলা health"
    assert t.decode(t.encode(s)) == s

def test_model_forward_and_generate():
    cfg={"vocab_size":259,"block_size":32,"n_layer":2,"n_head":2,"n_embd":32}
    m=AIBondhuGPT(cfg)
    x=torch.randint(0,259,(2,16))
    logits,loss=m(x,x)
    assert logits.shape==(2,16,259)
    assert loss is not None and torch.isfinite(loss)
    assert m.generate(x[:1,:4],max_new_tokens=2).shape==(1,6)
