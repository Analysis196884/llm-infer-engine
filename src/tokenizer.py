from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def encode(self, text: str, add_special_tokens: bool = True):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list):
        return self.tokenizer.decode(ids, skip_special_tokens=True)