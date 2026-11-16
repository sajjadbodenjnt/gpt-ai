import torch
from torch.utils.data import Dataset
from .tokenizer import ByteTokenizer

class TextDataset(Dataset):
    def __init__(self, path, block_size):
        self.block_size = block_size
        tok = ByteTokenizer()
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text:
                        chunks.extend(tok.encode(text + "\n"))
        self.tokens = torch.tensor(chunks, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.block_size)

    def __getitem__(self, index):
        start = index * self.block_size
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y
