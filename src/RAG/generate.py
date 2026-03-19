import pandas as pd
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import llm
import sys
import json
import pickle
from pathlib import Path


def load():
    script_dir = Path(__file__).parent
    
    index_path = script_dir / "eecs_ind.faiss"
    json_path = script_dir / "data_storage.json"
    bm25_path = script_dir / "bm25.pkl"
    df = pd.read_json(str(json_path), orient="records")
    
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    ind = faiss.read_index(str(index_path))
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    
    return model, ind, df, bm25

def get_context(q, model, index, df, bm25, k_dense=8, k_sparse=4):
    q_embed = model.encode([q]).astype('float32')
    faiss.normalize_L2(q_embed)
    _, fi = index.search(q_embed, k_dense)
    first = [i for i in fi[0] if i != -1]

    t_query = [w.lower() for w in q.split() if len(w) > 2]
    if not t_query:
        t_query = q.lower().split() 
        
    second = np.argsort(bm25.get_scores(t_query))[::-1][:k_sparse].tolist()
    return "\n\n".join([df.iloc[i]['txt'] for i in list(set(first + second))])


def main():
    q_path = sys.argv[1]
    pred_path = sys.argv[2]
    model, ind, df, bm25 = load()

    with open(q_path, 'r', encoding='utf-8') as f:
        questions = [line.strip().strip('"\'') for line in f if line.strip()]

    ans = []
    sys_prompt = """You are a highly precise QA bot for the UC Berkeley EECS department.
    Base your answer STRICTLY on the provided context. Follow these rules exactly:
    1. Output ONLY the answer — no explanation, no punctuation, no trailing text.
    2. If the question asks for a NAME, output ONLY the name. Never output a year, date, or number when a name is asked for.
    3. If the question asks for a PLACE, output ONLY the place name with NO extra details.
    4. If the question asks for a DATE or DEADLINE, output only the date in the exact format found in the context.
    5. If the question requires COUNTING items listed in the context, count them carefully and output only the number.
    6. If the question is Yes/No, output exactly "Yes" or "No".
    7. Never add parenthetical details, date ranges, or qualifiers after your answer.
    8. If the answer is not explicitly in the context, make your best guess based on any related information in the context or your base knowledge. When answering according to this rule, you must still follow the previous rules listed above."""
    
    for q in questions:
        try:
            context = get_context(q, model, ind, df, bm25)
            query = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
            res = llm.call_llm(query, sys_prompt, "meta-llama/llama-3.1-8b-instruct", 20, 0.0, 25)
            clean_res = res.replace("\n", " ").replace("\r", " ").strip()
            ans.append(clean_res)
        except Exception: # OpenRouter time-out check
            print(f"error with question: '{q}'")
            ans.append("not available")


    with open(pred_path, 'w', encoding='utf-8') as f:
        for a in ans:
            f.write(f"{a}\n")

if __name__ == "__main__":
    main()