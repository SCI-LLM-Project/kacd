#!/bin/bash
python -m outlines.serve.serve --model="mistralai/Mistral-7B-Instruct-v0.3" --dtype="float16" --tensor-parallel-size 2 --max-num-seqs 1 --seed 42
