import numpy as np
import pandas as pd
import os 

import ast
from functools import reduce

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
'''
Version: 2.0.1 
Date: 10.12.2023
'''
'''
Utility Functions 
'''

'''
function to read data from source 
name - str - name of the file 

returns
file - pandas Dataframe - file in pandas Dataframe form 
'''
def read_data(name):
    file = pd.read_csv('./data/id_' + name + '_mmsr.tsv', sep='\t')
    return file

'''
function to process a query and return the id 
song - str - song name from a query
artist - str - artist name from a query
info - pandas Dataframe - information of the songs 

returns - str 
id - id of the song 
'''
def get_id_from_info(song, artist, info):
    song_entry = info[(info['song'] == song) & (info['artist'] == artist)]
    
    if not song_entry.empty:
        id = song_entry.iloc[0]['id']
        return id
      
'''
function to display the result from dictionary
ids - list[str] - list which stores ids of the retrieved songs
info - pandas Dataframe - information of the songs
'''
def display_res(ids, info):
    trl = get_info_from_ids(ids, info)
    display_track_list(trl)

'''
function to get the names and artists from the ids of the retrieved tracks 
ids - List[str] - list which stores ids of the retrieved songs
info - pandas Dataframe - information of the songs

res - list[str] - list which stores the names and artists from the ids of the retrieved tracks 
'''
def get_info_from_ids(ids, info):
    res = []
    for id in ids:
        entry = info[info['id'] == id]
        if not entry.empty:
            res.append((entry.iloc[0]['song'], entry.iloc[0]['artist']))
    return res

'''
function to print the info from a list
trl - list((str, str)) - list containing info stored in tuple (name, artist)
'''
def display_track_list(trl):
    for tr in trl:
        print(f"Name: {tr[0]:<40} Singer: {tr[1]}")

'''
retrieval system for a representation 
id - str - id of the song in the query
repr - pandas Dataframe - representation of lyrics
N - int - number of retrieved tracks  
sim_func - func - similarity function 

returns - nd.array dtype=str
track_ids - ids of tracks retrieved 
'''
def text_based(id, repr, N, sim_func):

    # search for the row of the query song in the representation
    query_row = repr[repr['id'] == id]

    # exclude the id and index column
    query_vec = query_row.iloc[:, 2:].values[0]

    similarities = []

    # iterate through all tracks in the dataset
    for _ , row in repr.iterrows():
        track_vec = row.iloc[2:].values  #start from third column
        similarity = sim_func(query_vec, track_vec)
        similarities.append((row['id'], similarity))

    # Sort tracks by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Retrieve the N most similar tracks
    most_similar_tracks = similarities[1:N+1]

    # Retrieve the id of N most similar tracks
    res = [id for id, _ in most_similar_tracks]
    res = np.asarray(res)

    return res 

'''
return N tracks randomly
id - str - id of the song in the query
N - int - number of retrieved tracks
info - pandas Dataframe - information of the songs

returns - nd.array dtype=str
res - ids of tracks retrieved 
'''
def random_baseline(id, info, N):
    # Shuffle the songs DataFrame to get a random order
    shuffled_songs = info.sample(frac=1)
    
    # Exclude the query track using its ID
    shuffled_songs = shuffled_songs[shuffled_songs['id'] != id]
    
    # Select the top N rows as the retrieved tracks
    retrieved_tracks = shuffled_songs.head(N)

    # Get the id from these rows 
    res = retrieved_tracks['id'].tolist()
    res = np.asarray(res)

    return res

'''
wrapper function for cosine_similarity function to accept two numpy arrays 
arr1 - np.array - first input array
arr2 - np.array - second input array

returns - float
res - cosine similarity score of 2 functions 
'''
def cos_sim(arr1, arr2):
    arr1_reshape = arr1.reshape(1, -1)
    arr2_reshape = arr2.reshape(1, -1)
    res = cosine_similarity(arr1_reshape, arr2_reshape)[0][0]
    return res

'''
wrapper function for euclidean_distances function to accept two numpy arrays 
arr1 - np.array - first input array
arr2 - np.array - second input array

returns - float
res - euclidean similarity score of 2 functions 
'''

def euc_sim(arr1, arr2):
    arr1_reshape = arr1.reshape(1, -1)
    arr2_reshape = arr2.reshape(1, -1)
    res = euclidean_distances(arr1_reshape, arr2_reshape)[0][0]
    return res

"""
audio base retriever system (by Li)
id - str - id of the song in the query
repr - pandas Dataframe - representation of lyrics
N - int - number of retrieved tracks  
sim_func - func - similarity function 

returns - nd.array dtype=str
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


"""
genre coverage @ 10
genres - pd.DataFrame - genre data set 
query_id - str - query id 
retrieved_ids - List[str] - id of the retrieved tracks 

return: 
res - float - genre coverage @ 10 score


dependency: import numpy as np 
            import pandas as pd 
"""

def gen_cov_10(retrieved, genres):
    # 1.return number of unique genre in the dataset (offline, need optimization)
    # 1.1 convert all the values in column "genre" from str to nd.array
    genres["genre_arr"] = genres["genre"].apply(lambda x: np.array(ast.literal_eval(x)))
    
    # 1.2 return the union of all genres
    all_genres = reduce(np.union1d, genres["genre_arr"])
    num_all_genres = len(all_genres)
    
    # 2.return number of unique genre in the retrieved 
    # 2.1 return genre of queries in genre with id as index 
    retrieved_df = genres.loc[genres["id"].isin(retrieved.flatten())]
    
    # 2.2 return the union of all genres in queries 
    retrieved_genres = reduce(np.union1d, retrieved_df["genre_arr"]) 
    num_retrieved_genres = len(retrieved_genres)
    
    # 3. calculate the genre coverage@10
    res = num_retrieved_genres / num_all_genres 
    return res


"""
ndcg@10 score
query_id - str - query id 
retrieved_ids - List[str] - id of the retrieved tracks 
genres - pd.DataFrame - genre dataset 


return:
ndcg - float - ndcg@10 score 

"""

def ndcg_score(query_id, retrieved_ids, genres):
    # 1. convert all the values in column "genre" from str to nd.array
    genres["genre_arr"] = genres["genre"].apply(lambda x: np.array(ast.literal_eval(x)))
    
    # 2. calculate the rel for each track 
    query_genre = genres.loc[genres["id"] == query_id, 'genre_arr'].to_numpy()[0]
    retrieved_genre = pd.DataFrame(retrieved_ids, columns=['id'])
    retrieved_genre = pd.merge(genres, retrieved_genre, on="id", how="right")
    retrieved_genre["rel"] = retrieved_genre["genre_arr"].apply(lambda x: 2 * len(np.intersect1d(x, query_genre)) / (len(x) + len(query_genre)))
    
    # 3. calculate dcg
    rel = retrieved_genre["rel"].to_numpy()
    gain = np.empty(rel.shape)
    for i, _ in enumerate(rel):
        gain[...,i] = rel[...,i] / np.log2(i + 2)
    dcg = np.sum(gain)
    
    # 4. calculate idcg
    rel_sort = np.sort(rel)[::-1]
    rel_sort_gain = np.empty(rel_sort.shape)

    for i, _ in enumerate(rel_sort):
        rel_sort_gain[...,i] = rel_sort[...,i] / np.log2(i + 2)
    idcg = np.sum(rel_sort_gain)
    
    # 5. calculate ndcg
    ndcg = dcg / idcg
    return ndcg 