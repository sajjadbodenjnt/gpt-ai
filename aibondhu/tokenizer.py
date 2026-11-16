"""A reversible byte tokenizer. IDs 0-255 are UTF-8 bytes; 256-258 are reserved."""
class ByteTokenizer:
    vocab_size = 259
    def encode(self, text: str):
        return list(text.encode("utf-8"))
    def decode(self, ids):
        raw = bytes([int(i) for i in ids if 0 <= int(i) <= 255])
        return raw.decode("utf-8", errors="replace")
