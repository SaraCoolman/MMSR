import numpy as np
import pandas as pd
import statistics as st
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
'''
Version: 2.0.1 
Date: 26.11.2023
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
function to get the ids and genres from the ids of the retrieved tracks 
ids - list[str] - list which stores ids of the retrieved songs
info - pandas Dataframe - information of genres

res - list[str] - list which stores the genres from the ids of the retrieved tracks 
'''
def get_genre_from_ids(ids, genres):
    res = []
    for id in ids:
        entry = genres[genres['id'] == id]
        if not entry.empty:
            res.append((entry.iloc[0]['id'],entry.iloc[0]['genre']))
    return res


'''
function to get the id and genres from the id of the query track
info - pandas Dataframe - information of genres
res - list[str] - list which stores the genres from the id of the query track
'''
def get_genre_from_query(query_id, genres):
    entry = genres[genres['id'] == query_id]
    if not entry.empty:
        res = [(entry.iloc[0]['id'], entry.iloc[0]['genre'])]
    else:
        res = []
    return res

'''
function to calculate precision 
'''
    
def calculate_precision(query_genre, retrieved_genres):
    query_genres = set(eval(query_genre[0][1]))
    N = len(retrieved_genres)
    count = 0
    for song_id, genres_str in retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            count += 1
    precision = count / N
    return precision


def calculate_precision_at_k(query_genre, retrieved_genres, k):
    query_genres = set(eval(query_genre[0][1]))

    # Take only the top k items from retrieved_genres
    top_k_retrieved_genres = retrieved_genres[:k]

    count = 0
    for song_id, genres_str in top_k_retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            count += 1

    precision_at_k = count / k
    return precision_at_k



'''
function that counts relevant songs according to the query song in the whole dataset
'''

def count_relevant_songs_in_dataset(query_genre, dataset_genres):
    query_genres = set(eval(query_genre[0][1]))
    count = 0
    for song_id, genres_str in dataset_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            count += 1
    count
    return count

'''
function that counts relevant songs according to the query song in the retrieved result
'''
def count_relevant_songs_in_result(query_genre, retrieved_genres):
    query_genres = set(eval(query_genre[0][1]))
    count = 0
    for song_id, genres_str in retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            count += 1
    count
    return count


'''
function that calculates the recall
'''
def calculate_recall(query_genre, retrieved_genres, dataset_genres):
    query_genres = set(eval(query_genre[0][1]))
    relevant_retrieved_songs = 0
    for song_id, genres_str in retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            relevant_retrieved_songs += 1
    
    query_genres = set(eval(query_genre[0][1]))
    relevant_songs_dataset = 0
    for song_id, genres_str in dataset_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            relevant_songs_dataset += 1
            
    return  relevant_retrieved_songs / relevant_songs_dataset



'''
function that calculates the recall at k
'''
def calculate_recall_at_k(query_genre, retrieved_genres, dataset_genres, k):
    query_genres = set(eval(query_genre[0][1]))
    
    # Consider only the top k retrieved songs
    top_k_retrieved_genres = retrieved_genres[:k]

    relevant_retrieved_songs = 0
    for song_id, genres_str in top_k_retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            relevant_retrieved_songs += 1
    
    query_genres = set(eval(query_genre[0][1]))
    relevant_songs_dataset = 0
    for song_id, genres_str in dataset_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            relevant_songs_dataset += 1
            
    return relevant_retrieved_songs / relevant_songs_dataset

'''
function to calculate the average precision of all 3 query songs
'''
def average_precision(p1, p2, p3):
    precisions = [p1,p2,p3]
    average_precision = st.mean(precisions)
    return average_precision



'''
function to calculate the average recall of all 3 query songs
'''
def average_recall(r1, r2, r3):
    recalls = [r1,r2,r3]
    average_recall = st.mean(recalls)
    return average_recall


'''
function to plot the precision-recall curve
'''
def plot_precision_recall_curve(system_data):
    k_values = list(range(1, 100))

    plt.figure()

    for system_name, system_info in system_data.items():
        precisions = []
        recalls = []

        for k in k_values:
            precision = calculate_precision_at_k(system_info["query_genre"], system_info["retrieved_genres"], k)
            recall = calculate_recall_at_k(system_info["query_genre"], system_info["retrieved_genres"], system_info["dataset_genres"], k)

            precisions.append(precision)
            recalls.append(recall)

        plt.plot(recalls, precisions, label=system_info["system_name"])


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Evaluated Systems")
    plt.legend()


    plt.show()
    
    

'''
function to compute the genre distribution and shannons entropy (Sara´s function)
'''

def compute_genre_distribution(retrieved_result, dataset_genres):
    # Get unique genres in the dataset
    all_genres = set()
    for _, genres_str in dataset_genres:
        all_genres.update(eval(genres_str))

    # Initialize genre distribution vector with zeros
    genre_distribution = {genre: 0.0 for genre in all_genres}

    # Update genre distribution based on the retrieved tracks
    total_tracks = len(retrieved_result)
    for _, retrieved_genres_str in retrieved_result:
        retrieved_genres = set(eval(retrieved_genres_str))
        for genre in retrieved_genres:
            genre_distribution[genre] += 1.0 / total_tracks if total_tracks > 0 else 0.0

    # Convert the genre distribution to a list
    genre_distribution_list = [genre_distribution[genre] for genre in all_genres]
    
     # Normalize the genre distribution
    normalized_distribution = {genre: count / total_tracks for genre, count in genre_distribution.items()}

    # Calculate Shannon's entropy for genre diversity@10
    entropy = -sum(p * math.log2(p) for p in normalized_distribution.values() if p > 0)

    return normalized_distribution, entropy



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