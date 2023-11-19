"""
audio base retriever system (by Li)
id - str - id of the song in the query
repr - pandas Dataframe - representation of lyrics
N - int - number of retrieved tracks  
sim_func - func - similarity function 

returns - list[str]
track_ids - ids of tracks retrieved 
"""
import pandas as pd
import numpy as np 

def audio_based(id, repr, N, sim_func):
    # return the query song's row in repr
    target_row = repr[repr['id'] == id].iloc[:, 2:].to_numpy()
    # calculate similarity score
    repr['sim_score'] = repr.apply(lambda x:sim_func(x[2:].to_numpy(),target_row), axis=1)
    # sort tracks by similarity 
    sorted_repr = repr.sort_values(by='sim_score', ascending=False)
    # get the N most similar tracks 
    res = sorted_repr.iloc[1: N+1]['id'].to_numpy()
    return res 