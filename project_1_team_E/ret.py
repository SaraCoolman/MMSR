import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
'''
Version: 1.0.1 
Date: 14.11.2023
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
ids - list[str] - list which stores ids of the retrieved songs
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

returns - list[str]
track_ids - ids of tracks retrieved 
'''
def text_based(id, repr, N, sim_func):

    #search for the row of the query song in the representation and get the vector of the query song
    query_row = repr[repr['id'] == id]
    query_vec = query_row.iloc[:, 2:].values[0]

    similarities = []

    #iterate through all tracks in the representation dataset, compute similarity score, add song IDs and store to a list
    for _ , row in repr.iterrows():
        track_vec = row.iloc[2:].values
        similarity = sim_func(query_vec, track_vec)
        similarities.append((row['id'], similarity))

    #sort by similarity score from most similar to least similar and save N most similar tracks and retrieve ids
    similarities.sort(key=lambda x: x[1], reverse=True)
    most_similar_tracks = similarities[1:N+1]
    res = [id for id, _ in most_similar_tracks]

    return res 

'''
return N tracks randomly
id - str - id of the song in the query
N - int - number of retrieved tracks
info - pandas Dataframe - information of the songs

returns - list[str]
res - ids of tracks retrieved 
'''

def random_baseline(id, info, N):
    #put songs in random order and retrieve all shuffled songs without the query song
    shuffled_songs = info.sample(frac=1)
    shuffled_songs = shuffled_songs[shuffled_songs['id'] != id]
    
    #select top N songs from the shuffled dataset and save the id of the result to a list
    retrieved_tracks = shuffled_songs.head(N)
    res = retrieved_tracks['id'].tolist()
    return res

'''
wrapper function for cosine_similarity function to accept two numpy arrays 
arr1 - np.array - first input array
arr2 - np.array - second input array


returns - float
res - cosine similarity score of 2 functions 
'''

def cos_sim(arr1, arr2):
    #arrays need to be reshaped to 2d arrays
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