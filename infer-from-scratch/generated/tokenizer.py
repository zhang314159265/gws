from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    def __init__(self, path):
        self.tokenizer = HFTokenizer.from_file(path)
        self.im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

    def encode_chat(self, prompt):
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return self.tokenizer.encode(text).ids

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def is_end_token(self, t):
        return t in [self.eos_id, self.im_end_id]
