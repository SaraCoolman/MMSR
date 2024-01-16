import numpy as np
import pandas as pd
import math
import os 
import matplotlib.pyplot as plt
import ast
from functools import reduce
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve
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
    # Convert the repr dataframe to a numpy array, excluding the id column
    repr_array = repr.iloc[:, 2:].to_numpy()

    # Get the target row as a numpy array
    target_row = repr_array[repr['id'] == id]

    # Calculate the similarity score for all rows using broadcasting and matrix multiplication
    sim_score = sim_func(repr_array, target_row)

    # Assign the sim_score array to a new column in the repr dataframe
    repr['sim_score'] = sim_score
    # Sort the sim_score array in descending order and get the indices
    sorted_indices = np.argsort(-sim_score)

    # Get the N most similar tracks using numpy indexing
    res = repr['id'].to_numpy()[sorted_indices[1:N+1]]
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
def gen_div_10(genres_retrieved, all_genres, N):
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
function that calculates the recall at k
'''
'''def calculate_recall_at_k(query_genre, retrieved_genres, dataset_genres, k):
    query_genres = set(eval(query_genre[0][1]))
    
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
import numpy as np
import ast

def calculate_recall_at_k_vectorized(query_genre, retrieved_genres, dataset_genres, k):
    # Convert the input strings to arrays of sets
    # Use a list comprehension to get only the genres
    dataset_genres = [sublist[1] for sublist in dataset_genres]
    query_genre = [sublist[1] for sublist in query_genre]
    retrieved_genres = [sublist[1] for sublist in retrieved_genres]
    print("here")
    print(query_genre[0][1])
  
    print(retrieved_genres[1])
    query_genres = [set(ast.literal_eval(g)) for g in query_genre]
    retrieved_genres = [set(ast.literal_eval(g)) for g in retrieved_genres]
    dataset_genres = [set(ast.literal_eval(g)) for g in dataset_genres]
    print(query_genres)
    # Get the top k retrieved genres
    top_k_retrieved_genres = retrieved_genres[:k]
    
    # Define a function that checks if two sets have any common elements
    def has_common_elements(x, y):
        return len(x.intersection(y)) > 0
    
    # Vectorize the function to apply it to arrays of sets
    has_common_elements = np.vectorize(has_common_elements, otypes=[bool])
    
    # Count the relevant retrieved songs by applying the function to each row of query genres and top k retrieved genres
    relevant_retrieved_songs = np.sum(has_common_elements(query_genres, top_k_retrieved_genres))
    print(relevant_retrieved_songs)
    # Count the relevant songs in the dataset by applying the function to each row of query genres and dataset genres
    relevant_songs_dataset = np.sum(has_common_elements(query_genres, dataset_genres))
    print(relevant_songs_dataset)
    # Return the recall at k as an array
    return relevant_retrieved_songs / relevant_songs_dataset

'''
function to calculate precision @k
'''

def calculate_precision_at_k(query_genre, retrieved_genres, k):
    query_genres = set(eval(query_genre[0][1]))

    top_k_retrieved_genres = retrieved_genres[:k]

    count = 0
    for song_id, genres_str in top_k_retrieved_genres:
        genres = set(eval(genres_str))
        if any(genre in genres for genre in query_genres):
            count += 1

    precision_at_k = count / k
    return precision_at_k



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
    

def get_avg_precision_at_k(df, genres, dataset_genres, k):
    # for each query(=row in dataframe) do
    # find the genres of the query with get_genre_from_query(query_id, genres)
    # calculate precision at k=10 
    df['PrecisionAtK'] = df.apply(lambda row: calculate_precision_at_k(get_genre_from_query(row['id'], genres),  get_genre_from_ids(audio_based(row["id"], repr=df, N=100, sim_func=cos_sim), genres), 10), axis=1)

    # Calculate mean precision at k
    avg_precision = df['PrecisionAtK'].mean()

    return avg_precision    



def get_avg_recall_at_k(repr, N, genres):
    # for each query(=row in dataframe) do
    # find the genres of the query with get_genre_from_query(query_id, genres)
    # calculate recall at k=10 
    #audio_based(id=id_track1, repr=blf_correlation, N=10, sim_func=cos_sim)
    #print(genres)
 
    
    # Convert the repr dataframe to a numpy array, excluding the id column
    repr_array = repr.iloc[:, 2:].to_numpy()
    
    # Loop over all the rows in the repr dataframe
    # Define a helper function that takes a row of the repr dataframe as input and returns the recall at k
    def helper(row):
        print(1)
        # Get the query id and vector as numpy arrays
        query_id = row['id']
        query_vector = row[2:].to_numpy()

        cos_sim_matrix = np.triu(repr_array.dot(query_vector) / (np.linalg.norm(repr_array, axis=1) * np.linalg.norm(query_vector)))
        
        # Calculate the cosine similarity matrix using numpy dot product and broadcasting, but only for the upper triangular part
        ##cos_sim_matrix = np.triu(repr_array.dot(query_vector) / (np.linalg.norm(repr_array, axis=1) * np.linalg.norm(query_vector)))
        
        # Fill the lower triangular part with the same values as the upper triangular part
        cos_sim_matrix = cos_sim_matrix + cos_sim_matrix.T - np.diag(cos_sim_matrix.diagonal())
        
        # Sort the cos_sim_matrix in descending order and get the indices
        sorted_indices = np.argsort(-cos_sim_matrix)
        
        # Get the N most similar tracks using numpy indexing, excluding the query itself
        #print(genres)
        res = repr['id'].to_numpy()[sorted_indices[1:11]]
        query_genres = get_genre_from_query(query_id, genres)
        retrieved_genres = get_genre_from_ids(res,genres)
        dataset_genres = get_genre_from_ids(repr["id"],genres)
        
        repr["RecallAtK"] =calculate_recall_at_k_vectorized(query_genres,retrieved_genres , dataset_genres, 10)
        #print(index)
        # Return the recall at k
        return recall_at_k
    # Apply the helper function to each row of the repr dataframe using pandas.DataFrame.apply
    repr["RecallAtK"] = repr.apply(helper, axis=1)
    #df['RecallAtK'] = df.apply(lambda row: calculate_recall_at_k(get_genre_from_query(row['id'], genres),get_genre_from_ids(audio_based(row["id"], repr=df, N=100, sim_func=cos_sim),genres),dataset_genres,10), axis=1)
    # Apply the vectorized function with k = 2
    #df['RecallAtK'] = df.apply(lambda row: calculate_recall_at_k_vectorized([row["query_genre"]], [row["retrieved_genres"]], [row["dataset_genres"]], 10), axis=1)


    # Calculate mean recall at k
    avg_recall = repr['RecallAtK'].mean()

    return avg_recall

def get_relevancy_count(genre_query,genre_other_songs):
    array = change_list_to_sets(genre_other_songs)



    # Initialize a counter
    count = 0

    for s in array:
        # Check if the intersection is not empty
        if genre_query.intersection(s):#
        # Increment the counter
            count += 1

    print("done")
    #print(count) # 1
    return count
def change_list_to_sets_2(list):
    array = set()

    for s in list:
        s =change_to_set(s)
        # Fügen Sie das Set zum Array hinzu
        array.update(s)
        #print(array)
        #print(" ")
    return array
def change_list_to_sets(list):
    array = []
    
    for s in list:
        s =change_to_set(s)
        # Fügen Sie das Set zum Array hinzu
        array.append(s)
        #print(array)
        #print(" ")
    return array
def generate_cos_sim_matrix(embedding):
    print(embedding.shape)
   
    
    #embedding_matrix,embedding_transpose = make2embeddings1(embedding1, embedding2)
    embedding_matrix,embedding_transpose = convert2matrix(embedding)

    #embedding_transpose = embedding_matrix.transpose()
    # Form der transponierten aMatrizen
    #print(embedding_matrix.shape) # (1328, 10093)
    #print(len(blf_correlation), len(blf_correlation_transpose_zip[0])) # (1328, 10093)
    # Berechnen Sie die Kosinusähnlichkeitsmatrix
    cos_sim_matrix = np.triu(embedding_matrix.dot(embedding_transpose) / (np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(embedding_matrix, axis=1)[:, None]))
    # Fill the lower triangular part with the same values as the upper triangular part
    cos_sim_matrix = cos_sim_matrix + cos_sim_matrix.T - np.diag(cos_sim_matrix.diagonal())

    # Create a dataframe from the numpy matrix
    df = pd.DataFrame(cos_sim_matrix)

    # Assign the ids to the index and columns
    df.index = embedding.iloc[:, 0]
    #df.index = df.index.str.strip('(),')
    df.columns = embedding.iloc[:, 0]
    #df.columns = df.columns.str.strip('(),')

    return df
def make2embeddings1(embedding1, embedding2):
    #Ziel der Function: early data fusion und anschließend dim reduction mit pca
    embedding_array = data_fusion(embedding1.head(10094),embedding2 )
    #Apply UMAP
    # Convert to NumPy array
    embedding_array = np.asarray(embedding_array)
    #umap_model = umap.UMAP(n_neighbors=4, min_dist=0.8, n_components=100)  # Adjust parameters as needed
    #print("Starting UMAP")
    #embedding_umap = umap_model.fit_transform(embedding_array)
    #print("Stopping UMAP")
    # Convert the UMAP result to a matrix
    #embedding_matrix = np.matrix(embedding_umap)
    
    # Apply PCA
    pca_model = PCA(n_components=100)  # Adjust the number of components as needed
    print("Starting PCA")
    embedding_pca = pca_model.fit_transform(embedding_array)
    print("Stopping PCA")
    
    # Convert the PCA result to a matrix
    embedding_matrix = np.matrix(embedding_pca)
    return embedding_matrix, embedding_matrix.transpose()
def convert2matrix(embedding):
    #this function prepares the calc of the cos-sim-similarity by making the encoding a matrix: reason is faster
    embedding = embedding.sort_values(by='id')
    # Konvertieren DataFrame in ein NumPy-ndarray
    embedding_array = embedding.iloc[:, 2:].to_numpy()
    # Konvertieren NumPy-array in eine NumPy-Matrix
    # initialize the LDA model 
    #lda = LatentDirichletAllocation(n_components=50, max_iter=5, learning_method='online', learning_offset=50., random_state=0)

    '''# Apply UMAP
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.2, n_components=50)  # Adjust parameters as needed
    print("Starting UMAP")
    embedding_umap = umap_model.fit_transform(embedding_array)
    print("Stopping UMAP")'''
    
    # Convert the UMAP result to a matrix
    embedding_matrix = np.matrix(embedding_array)
    
    # Transponieren Sie die Matrix mit der transpose() Methode
    #embedding_transpose = embedding_matrix.transpose()
    #embedding_mm_fused = data_fusion(tfidf[:10094], musicnn)
    
    # Transponieren 
    embedding_transpose = embedding_matrix.transpose()
    return embedding_matrix,embedding_transpose
    

   
   
  