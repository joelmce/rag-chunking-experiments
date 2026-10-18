import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    E = np.array(model.encode(texts, show_progress_bar=False))
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def sentence_splitter(text):
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$", text)]
    return [(s, e) for s, e in spans if text[s:e].strip()]

def fixed_chunks(text, size):
    return [{"start": s, "end": min(s + size, len(text)), "text": text[s:s + size]}
            for s in range(0, len(text), size)]

def combine_sentences(sentences, buffer):
    combined = [" ".join(sentences[max(0, i - buffer): i + buffer + 1])
                for i in range(len(sentences))]
    return combined

def semantic_chunks(text, buffer=1, percentile=95):
    spans = sentence_splitter(text)
    if len(spans) < 2:
        return [{"start": 0, "end": len(text), "text": text}] if spans else []

    sentences = [text[s:e].strip() for s, e in spans]
    combined = combine_sentences(sentences, buffer)
    E = embed(combined)
    distances = 1 - np.sum(E[:-1] * E[1:], axis=1)

    threshold = np.percentile(distances, percentile)
    breakpoints = np.where(distances > threshold)[0]

    chunks, start = [], 0
    for b in list(breakpoints) + [len(sentences) - 1]:
        s, e = spans[start][0], spans[b][1]
        chunks.append({"start": s, "end": e, "text": text[s:e]})
        start = b + 1
    return chunks

text = Path("./data/coffee.txt").read_text(encoding='utf-8')
chunks = semantic_chunks(text)
print(len(chunks), [len(c["text"]) for c in chunks])

    
