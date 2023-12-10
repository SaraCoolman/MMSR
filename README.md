## Version 2.0.1
10.12.2023

## Description
Project created for Task 1 of WS2023/2024 Multimedia Search and Retrieval 
Contains a text based retrieval system which support tsv format files with tfidf/bert/word2vec word embeddings and cosine/euclidean similarity functions.  
Also contains a baseline retrieval system which return random tracks

## Changes
1. Added four audio based retrieval systems (ivec256, mfcc_stats, musicnn, blf_correlation)
2. Enabled evaluation (precision, recall, genre_coverage, genre_diversity, ndcg)

## Bug fix 
1. Changed the type of return values of following functions from List(str) to ndarray, dtype=str
    random_baseline(id, info, N) 
    text_based(id, repr, N, sim_func)

## Instruction 
1. put the data files in the data folder following the naming convention in the project structure 
2. launch main.ipynb and follow the instruction

## Project Structure
project_1_team_E
|-data
|   |-id_blf_correlation_mmsr.tsv
|   |-id_genres_mmsr.tsv
|   |-id_information_mmsr.tsv
|   |-id_ivec256_mmsr.tsv
|   |-id_lyrics_bert_mmsr.tsv
|   |-id_lyrics_tf-idf_mmsr.tsv
|   |-id_lyrics_word2vec_mmsr.tsv
|   |-id_mfcc_stats_mmsr.tsv
|   |-id_musicnn_mmsr.tsv
|-main.ipynb
|-README.md
|-result_mod.json
|-ret.py

main.ipynb: user interface 
README.md: readme file 
result_mod.json: json file to generate empty dictionary to store result
ret.py: python file for retrieval systems and utility functions 