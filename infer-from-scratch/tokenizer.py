import base64
import regex

DEBUG = False

def load_tokenizer(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            if len(parts) != 2:
                continue
            wordstr, rank = parts
            rank = int(rank)
            word = base64.b64decode(wordstr)
            out[word] = rank
    return out

class Tokenizer:
    PATTERN_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, path):
        self.bytes2rank = load_tokenizer(path)
        self.bos = 128000
        self.eos = 128001
        self.eot = 128009
        self.bytes2rank[b"<|begin_of_text|>"] = self.bos

        self.rank2bytes = {rank: _bytes for _bytes, rank in self.bytes2rank.items()}
        print(f"{len(self.bytes2rank)=}")


    def encode_segment(self, text):
        if DEBUG:
            print(f"encode segment: '{text}'")
        parts = [bytes([b]) for b in text.encode("utf-8")]

        while len(parts) >= 2:
            best_rank = None
            best_idx = 0
            for idx, (lhs, rhs) in enumerate(zip(parts[:-1], parts[1:])):
                combine = lhs + rhs
                rank = self.bytes2rank.get(combine)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = idx
            if best_rank is None:
                break
            parts = parts[:idx] + [parts[idx] + parts[idx + 1]] + parts[idx + 2:]
   
        if DEBUG:
            for part in parts:
                print(f" part: {part}")
        return [self.bytes2rank[part] for part in parts]

    def encode(self, text, bos=True):
        out = []
        if bos:
            out.append(self.bos)

        for match in regex.finditer(self.PATTERN_STR, text):
            out += self.encode_segment(match.group())
        return out

    def decode(self, tokens):
        allbytes = []
        for t in tokens:
            try:
                allbytes.extend(list(self.rank2bytes[t]))
            except:
                breakpoint()
                raise
        return bytes(allbytes).decode("utf-8")

    def is_end_token(self, t):
        return t in [self.eos, self.eot]

if __name__ == "__main__":
    tokenizer = Tokenizer("artifact/meta-llama/Meta-Llama-3-8B-Instruct/original/tokenizer.model")
    prompt = "Show me the C code for bubble sort."
    tokens = tokenizer.encode(prompt, bos=False)
    assert tokens == [7968, 757, 279, 356, 2082, 369, 24529, 3460, 13], f"Got {tokens}"
    prompt_back = tokenizer.decode(tokens)
    assert prompt == prompt_back, f"{prompt} v.s. {prompt_back}"
    print("PASS")
