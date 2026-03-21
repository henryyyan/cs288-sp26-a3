#!/bin/bash
cat data_storage_100-60.json.part.* > src/RAG/data_storage_100-60.json
cat eecs_ind_100-60.faiss.part.* > src/RAG/eecs_ind_100-60.faiss
python3 src/RAG/generate.py "$1" "$2"
