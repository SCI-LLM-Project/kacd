# kg_aug_causal_disc_exp

# Instructions for use

This repository contains code for creating a Knowledge Graph using structured output from LLMs. 

For a runnable, end-to-end example of the whole pipeline (KG construction -> predictions -> consolidation -> directed results), see `wflow.ipynb` - it runs every script below in order and is the fastest way to understand how the pieces fit together.

# How to create graph
- start neo4j in docker
- python construct_kg.py
(a dump of the graph constructed and used is in /mnt/danderson/neo4j/dumps/)

# How to generate predictions
- Make sure LLM_API_KEY is set in .env, and config.py's LLM_MODEL points at the Together model/endpoint you want
- Make sure neo4j is running in docker
- python query_kg.py && python query_llm.py && python query_rag.py && python query_causal_lit_kg.py && python query_causal_lit_llm.py && python query_causal_lit_rag.py (order doesn't matter)
- run the "Combine all results" section of wflow.ipynb to create results/results_undirected_combined.csv
- run query_directions_without_proto.py and/or query_directions_with_proto.py
    - the proto model results are in `results/proto.csv`. This is used in `query_directions_with_proto.csv`

# how to dump a database

This creates a consistent backup file:

## Stop the database first
docker exec neo4j-kg neo4j stop

## Create the dump (this will create a file in the container)
docker exec neo4j-kg neo4j-admin database dump neo4j --to-path=/var/lib/neo4j/import

## Start the database again
docker exec neo4j-kg neo4j start

## Copy the dump file from container to your host
docker cp neo4j-kg:/var/lib/neo4j/import/neo4j.dump {file}.dump

The dump file will be saved to your current directory as {file}.dump.

## neo4j loading instructions
From neo4j-admin dump:
docker exec neo4j-kg neo4j stop
docker exec neo4j-kg neo4j-admin database load neo4j --from-path={file}
docker exec neo4j-kg neo4j start

## DUMPS
- all dumps are saved to /mnt/danderson/neo4j/dumps
- /mnt/danderson/neo4j/dumps/full_corpus.dump is a dump of a KG that used all papers in the most up to date version of Conor's decision log (as of dec 3 2025)
- /mnt/danderson/neo4j/dumps/december2.dump is missing 5 papers and contains 1 extra papers