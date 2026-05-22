import json
import regex

class Tokenizer:
    PATTERN_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def setup_byte2char(self):
        nextavail = 256
        self.byte2char = {}
        for idx in range(256):
            ch = chr(idx) 
            if not (ch.isprintable() and not ch.isspace()):
                ch = chr(nextavail)
                nextavail += 1
            self.byte2char[idx] = ch
        self.char2byte = {ch: bt for bt, ch in self.byte2char.items()}

    def __init__(self, tokenizer_path):
        self.setup_byte2char()

        with open(tokenizer_path) as f:
            config = json.load(f)

        self.charseq2id = config["model"]["vocab"]
        self.id2charseq = {idx: charseq for charseq, idx in self.charseq2id.items()}
       
        self.pair2rank = {}
        for rank, pairstr in enumerate(config["model"]["merges"]):
            pair = tuple(pairstr.split())
            self.pair2rank[pair] = rank

        special_tokens = {}  # text -> token id
        for meta in config["added_tokens"]:
            special_tokens[meta["content"]] = meta["id"]

        self.im_end_id = special_tokens["<|im_end|>"]
        self.eos_id = special_tokens["<|endoftext|>"]

    def encode_segment(self, text):
        parts = [self.byte2char[bt] for bt in text.encode("utf-8")]

        while len(parts) >= 2:
            best_rank = None
            best_pos = None

            for pos in range(len(parts) - 1):
                lhs, rhs = parts[pos: pos + 2]
                rank = self.pair2rank.get((lhs, rhs))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = pos

            if best_rank is None:
                break

            parts = parts[:best_pos] + [parts[best_pos] + parts[best_pos + 1]] + parts[best_pos + 2:]

        return [self.charseq2id[part] for part in parts]

    def encode(self, text):
        out = []
        for seg in regex.finditer(self.PATTERN_STR, text):
            out.extend(self.encode_segment(seg.group()))
        return out

    def decode(self, tokens):
        bytelist = []
        for tok in tokens:
            charseq = self.id2charseq[tok]
            for ch in charseq:
                bytelist.append(self.char2byte[ch])

        return bytes(bytelist).decode("utf-8")

    def is_end_token(self, tok):
        return tok in [self.eos_id, self.im_end_id]
