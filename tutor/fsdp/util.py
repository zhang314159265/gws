import hashlib

def compute_tensor_hash(t):
    hasher = hashlib.sha256()
    hasher.update(t.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:8]

