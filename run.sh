#!/bin/bash
cat data_storage_100-60.json.part.* > data_storage_100-60.json
cat eecs_ind_100-60.faiss.part.* > eecs_ind_100-60.faiss
python3 src/RAG/generate.py "$1" "$2"
