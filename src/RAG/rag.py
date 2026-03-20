import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import torch
from rank_bm25 import BM25Okapi

parsed_dir = Path("../crawler/parsed_documents")

device = "mps" if torch.backends.mps.is_available() else "cpu"
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

chunk_sizes = [100, 300]
overlaps = [20, 60]

for chunk_size in chunk_sizes:
    for overlap in overlaps:
        step = chunk_size - overlap
        data = []

        for file in parsed_dir.iterdir():
            if file.is_file():
                try:
                    text = file.read_text(encoding='utf-8')
                    words = text.split()
                    if not words:
                        continue

                    for i in range(0, len(words), step):
                        chunk_text = " ".join(words[i:i+chunk_size])
                        data.append({'txt': chunk_text})

                except Exception:
                    continue

        df = pd.DataFrame(data, columns=['txt'])
        name = f"{chunk_size}-{overlap}"

        embed = embed_model.encode(df['txt'].tolist(), show_progress_bar=True, batch_size=16)
        embed = np.array(embed).astype('float32')

        dim = embed.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embed)
        index.add(embed)

        faiss.write_index(index, f"eecs_ind_{name}.faiss")
        df.to_json(f"data_storage_{name}.json", orient="records")

        bm25 = BM25Okapi([[word.lower() for word in doc.split() if len(word) > 2] for doc in df['txt'].tolist()])
        with open(f"bm25_{name}.pkl", "wb") as f:
            pickle.dump(bm25, f)

        print(f"Done: chunk_size={chunk_size}, overlap={overlap}")