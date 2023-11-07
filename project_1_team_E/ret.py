import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

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
retrieval system for a representation 
id - str - id of the song in the query
repr - pandas Dataframe - representation of lyrics
N - int - number of retrieved tracks  
sim_func - func - similarity function 

returns - list[str]
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
    # Shuffle the songs DataFrame to get a random order
    shuffled_songs = info.sample(frac=1)
    
    # Exclude the query track using its ID
    shuffled_songs = shuffled_songs[shuffled_songs['id'] != id]
    
    # Select the top N rows as the retrieved tracks
    retrieved_tracks = shuffled_songs.head(N)

    # Get the id from these rows 
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