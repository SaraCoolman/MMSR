import numpy as np
import pandas as pd
import math
import ast
from functools import reduce
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

def genre_diversity_at_10(genres_retrieved, all_genres):
    N = len(all_genres)
    total_tracks = len(genres_retrieved)
    
    genre_distribution_sum = np.zeros(N)
    
    for track_genres in genres_retrieved:
        num_genres = len(track_genres)
        contribution = 1 / num_genres if num_genres > 0 else 0
        
        for genre in track_genres:
            genre_index = all_genres.index(genre)
            genre_distribution_sum[genre_index] += contribution
    
    # Normalize the distribution
    normalized_distribution = genre_distribution_sum / total_tracks
    
    # Calculate Shannon's entropy
    entropy = -np.sum(normalized_distribution * np.log2(normalized_distribution, where=(normalized_distribution != 0)))
    
    return entropy

"""
function to get the genre from ids
"""
def get_genre(id,genres_df):
  # print(genres_df[genres_df['id'] == id ]['id'].values[0],'--->',id)
  return set(genres_df[genres_df['id'] == id ]['genre'].values[0].replace("[", "").replace("]", "").replace("'", "").split(', '))


"""
'Paramters:'
'genres_retrived: (list(sets)--> [{},{}...]) list of sets of the genres of the retrived tracks/songs '
'all_genres: (list) of all unique genres in the whole dataset'
'N: (int) the number of retrived tracks/songs'
'returns: (float) the Genre diversity@N'
"""
#diversity function provided by Ali

def diversity(genres_retrieved, all_genres, N):
    zeros_vec = np.zeros(len(all_genres))
    
    for g in genres_retrieved:
        leng_g = len(g)
        
        for g_i in g:
            position = all_genres.index(g_i)
            g_i_contribution = 1 / leng_g
            zeros_vec[position] += g_i_contribution

    result_vec = zeros_vec / N
    
    # Shannon's Entropy Calculation:
    diversity_value = 0
    
    for item in result_vec:
        if item != 0:
            diversity_value += item * math.log(item, 2)
    
    return -diversity_value



"""
genre coverage @ 10
genres - pd.DataFrame - genre data set 
queries - nd.array - 2d array of results from query in ids 

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