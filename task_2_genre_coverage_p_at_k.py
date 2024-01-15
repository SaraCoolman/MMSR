import numpy as np
import pandas as pd
import ast
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
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


'''
return N tracks randomly
id - str - id of the song in the query
N - int - number of retrieved tracks
info - pandas Dataframe - information of the songs

returns - list[str]
res - ids of tracks retrieved 
'''
def text_based(id, repr, N, sim_func, genre_id_mapping):

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
    #print(similarities)
    most_similar_tracks = similarities[1:N+1]
    res = [id for id, _ in most_similar_tracks]
    #print("1")

    #print("genre_coverage")
    # print(genre_coverage(genre_id_mapping, similarities))
    

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

    

            
'''
calculate the precision@K and recall@KeyError
K is N in our case
N - int - number of retrieved tracks
similarities - list - list of song IDs 
genre_id_mapping - python pandas dataframe -maps song ID to list of the genres in the song

returns i dont know yet
'''
def precision_and_recall_at_k(N, repr, genre_id_mapping):
    #print(repr)
    for id_query in repr["id"]:
        
        # Check if the Series is not empty
        if not genre_id_mapping[genre_id_mapping['id'] == id_query]["genre"].empty:
            list_of_genres =  ast.literal_eval(genre_id_mapping[genre_id_mapping['id'] == id_query]["genre"].iloc[0])
            # print(list_of_genres)
       
        else:
            print(f"No genre information found for id {id_non_query}")
        print("genre list done")
        
        #res = text_based(id=id_query, repr=repr, N=N, sim_func=cos_sim, genre_id_mapping = genre_id_mapping)
        print("similarities computed")
        res_df = pd.DataFrame(res,columns =['id'])
        res_df["precision"] = 0
        res_df["recall"] = 0
        #print(res_df.columns)
        relevant = []
        for id_non_query in repr['id']:
           
            
            
            
            
               
                
            # Check if the Series is not empty
            if not genre_id_mapping[genre_id_mapping['id'] == id_non_query]["genre"].empty:
                list_of_genres_query = ast.literal_eval(genre_id_mapping[genre_id_mapping['id'] == id_non_query]["genre"].iloc[0])
                #print(list_of_genres_query)
               
            else:
                print(f"No genre information found for id {id_query}")
            

            
            
            if set(list_of_genres) & set(list_of_genres_query):
                relevant.append(1)
                #print("1")
            else:
                relevant.append(0)
                #print("0")
         
                    
        #print(relevant)
        i = 0
        sum_relevant_k = 0
        
        while i < N:
            #print(i)
            sum_relevant_k += relevant[i] 
            i = i+1
        recall_at_k = sum_relevant_k/sum(relevant)
        
        precision_at_k = sum_relevant_k/N
        print(precision_at_k)
        
    
    res_df.at[id_non_query, "recall"] = recall_at_k
    res_df.at[id_non_query, "precision"] = precision_at_k
        

    fig, ax = plt.subplots()
    # plot the precision-recall curve on the given axes
    ax.plot(res_df["recall"],res_df["precision"])
    # set the axes labels and title
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    plt.show()

    average_precision_at_k = res_df["precision"].mean()
    average_recall_at_k = res_df["recall"].mean()

    return average_precision_at_k, average_recall_at_k
    
    

    
    
# genre coverage

def genre_coverage(genre_id_mapping, similarities):
    retrieved = similarities[1:11]
    # Apply a lambda function to transform each element in the "genres" into a set and add it to the set "all_genres"
    all_genres_in_data = set()
    genre_id_mapping["genre"].apply(lambda genres: all_genres_in_data.update(set(list(genres))))
    nr_unique_genres_overall = len(all_genres_in_data)
    all_genres_retrieved = set()
    merged_on_id = pd.merge(genre_id_mapping, similarities, on="id")
    merged_on_id["genre"].apply(lambda genres: all_genres_retrieved.update(set(list(genres))))
    nr_unique_genres_retrieved = len(all_genres_retrieved)
    return nr_unique_genres_retrieved/nr_unique_genres_overall
    
    


        
            
            
            
    
        
   
        
        
        

        

    