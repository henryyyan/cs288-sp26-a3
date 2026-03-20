#!/bin/bash
cat src/RAG/data_storage_300-60.json.part.* > src/RAG/data_storage_300-60.json
cat src/RAG/eecs_ind_300-60.faiss.part.* > src/RAG/eecs_ind_300-60.faiss
python3 src/RAG/generate.py "$1" "$2"
