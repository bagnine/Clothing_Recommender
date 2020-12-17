import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import scipy.spatial.distance as dist
from sklearn.neighbors import NearestNeighbors
from collections import Counter

#================================================ K Means Clustering Recommender =========================================

def cluster_recommendations(input_row, dataframe, n_clusters):
    '''Recommender that uses clustering to recommend a subset of the data that is
       clustered with the target.
       
       Takes in positional arguments for input_row (one row from a dataframe symmetrical
       to the dataframe argument), a dataframe and a number of clusters to create.
       
       Returns a list of all indices from the dataframe associated with the target's cluster.'''
    
    # Instantiate and fit the model
    kmm = KMeans(n_clusters= n_clusters)
    kmm.fit(dataframe)

    # Create tuples of each item's index and the predicted cluster number
    clusters = list(zip(list(dataframe.index), kmm.predict(dataframe) ))
    
    # Create a counter to hold the cluster of the target
    target = 0
    rec_list = []
    for i in clusters:
        if i[0] == input_row.index[0]:
            target == i[1]
            
            # Stop once the target cluster is identified and saved
            break
    
    # Iterate through the indices and append the ones that belong to the target 
    # cluster to the list
    for i in clusters:
        if i[1] == target:
            rec_list.append(i[0])
    
    # Return the selected list of indices
    return rec_list

def display_kmeans_recs(input_row, dataframe, n_clusters):
    '''Takes in a target row, a dataframe and number of clusters. 
       Using KMeans, assigns each observation to clusters, then finds the cluster
       of the target row and displays the target and 5 random images from it.'''
    
    # Take indices from clusters with the target row
    indices = cluster_recommendations(input_row, dataframe, n_clusters)
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    images = []
    images.append(Image.open(f'./src/images/{input_row.index[0]}.jpg'))
    
    for i in np.random.choice(indices, size=6, replace=True):
        images.append(Image.open(f'./src/images/{i}.jpg'))
    
              
    # Display the images
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on clustering', size = 16)
    ax[0,0].imshow(images[0])
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(images[2])
    ax[0,1].axis('off')
    ax[0,1].set_title('Recommendation 1')
    ax[0,2].imshow(images[3])
    ax[0,2].axis('off')
    ax[0,2].set_title('Recommendation 2')
    ax[1,0].imshow(images[4])
    ax[1,0].axis('off')
    ax[1,0].set_title('Recommendation 3')
    ax[1,1].imshow(images[5])
    ax[1,1].axis('off')
    ax[1,1].set_title('Recommendation 4')
    ax[1,2].imshow(images[6])
    ax[1,2].axis('off')
    ax[1,2].set_title('Recommendation 5')
    plt.show()
    
    return 

#================================================ K Nearest Neighbors Recommender ========================================

def KNN_recs(input_row, dataframe, n_neighbors, n_recs = 8):
    '''Using K Nearest Neighbors, takes in a target row, a dataframe, and a
       specified number of k neighbors and the desired number of recommendations
       (default 8).
       
       Returns the recommendations.'''
    
    # Instantiate and fit Nearest Neighbors
    knn = NearestNeighbors(n_neighbors= n_neighbors, n_jobs=-1)
    knn.fit(dataframe)

    # Generate the closest rows
    recs = knn.kneighbors(X= input_row, n_neighbors=n_recs, return_distance=False)
    indices = []

    # Re-calculate to get the correct indices
    for i in recs[0]:
        indices.append(dataframe.iloc[[i]].index[0])
    return indices[1:]

def display_KNN_recs(input_row, dataframe, n_neighbors):
    '''Using KNN_recs as a helper function, displays the corresponding images of 5
       recommendations, plus the target'''
    # Take indices from nearest neighbors
    indices = KNN_recs(input_row, dataframe, n_neighbors)
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    images = []
    
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))
    
              
    # Display the images
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on Nearest Neighbors', size = 16)
    ax[0,0].imshow(images[0])
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(images[1])
    ax[0,1].axis('off')
    ax[0,1].set_title('Recommendation 1')
    ax[0,2].imshow(images[2])
    ax[0,2].axis('off')
    ax[0,2].set_title('Recommendation 2')
    ax[1,0].imshow(images[3])
    ax[1,0].axis('off')
    ax[1,0].set_title('Recommendation 3')
    ax[1,1].imshow(images[4])
    ax[1,1].axis('off')
    ax[1,1].set_title('Recommendation 4')
    ax[1,2].imshow(images[5])
    ax[1,2].axis('off')
    ax[1,2].set_title('Recommendation 5')
    plt.show()
    
    return 

#================================================ Cosine Similarity Recommender ==========================================

def cosine_similarity_recs(input_row, dataframe):
    '''Takes in a row of data and a symmetrical DataFrame, calculates the consine
       similarity of each row with the target and returns a list of the highest 
       scores and their indices'''
    
    # Create an empty list to populate with cosine similarities
    cos_sim = []
    
    # Iterate through every row of the dataframe and add its cosine similarity to the 
    # target row to the list as a tuple including the item's index
    for i in dataframe.index:
        try:
            cos_sim.append((cosine_similarity(input_row, dataframe.iloc[i].values.reshape(1,-1)), i))
        except IndexError:
            continue
    
    # Return a list of the closest cosine similarities and their indices
    return sorted(cos_sim)[-2:-11:-1]

def display_cosim_recs(input_row, dataframe):
    '''Using cosine_similarity_recs, generates recommendations based on a 
       target row, then displays the corresponding images of target and recs'''
    
    # Take indices from closest cosine similarities to the target row
    indices = [i[1] for i in cosine_similarity_recs(input_row, dataframe)]
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    images = []
    images.append(Image.open(f'./src/images/{input_row.index[0]}.jpg'))
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))
    
    # Display the images
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on Cosine Similarity', size = 16)
    ax[0,0].imshow(images[0])
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(images[2])
    ax[0,1].axis('off')
    ax[0,1].set_title('Recommendation 1')
    ax[0,2].imshow(images[3])
    ax[0,2].axis('off')
    ax[0,2].set_title('Recommendation 2')
    ax[1,0].imshow(images[4])
    ax[1,0].axis('off')
    ax[1,0].set_title('Recommendation 3')
    ax[1,1].imshow(images[5])
    ax[1,1].axis('off')
    ax[1,1].set_title('Recommendation 4')
    ax[1,2].imshow(images[6])
    ax[1,2].axis('off')
    ax[1,2].set_title('Recommendation 5')

    return plt.show()

#=============================================== Ensemble Recommender ====================================================

def ensemble_recs(input_row, dataframe, color_dataframe, n_clusters= 18, model_weights = None):
    '''Takes in a target input, dataframe, dataframe of RGB tuples, number of 
       clusters (default 18) and an optional model weight parameter.
       
       Uses KNN, clustering, Euclidean distance of color tuples and cosine 
       similarity to generate recommendation indices. 
       
       The indices are then added together. If 'visual' is selected in the 
       model_weight field, color distance is double weighted. If cosine is 
       selected, its recommendations are doubled. Otherwise all recommendations
       are weighted equally.
       
       The indices are then collated into a list of ten recommendations based on
       frequency between the four models and returned.'''
    
    cos_sim = []
    color_output = []
    
    # Generate 500 KNN recommendations
    knn = KNN_recs(input_row, dataframe, 8, n_recs = 500)
    
    # Generate ~500 recs based on KMeans clustering
    kmean = cluster_recommendations(input_row, dataframe, n_clusters=n_clusters)

    # Generate a list of nearest RGB distances
    for i in dataframe.index:
        try:
            color_output.append((sum(
                [dist.euclidean(color_dataframe['0'][input_row.index[0]], color_dataframe['0'][i]),
                dist.euclidean(color_dataframe['1'][input_row.index[0]], color_dataframe['1'][i]),
                dist.euclidean(color_dataframe['2'][input_row.index[0]], color_dataframe['2'][i])]), i))
        except KeyError:
            continue
    # Generate cosine similarity recommendations
        try:
            cos_sim.append((cosine_similarity(input_row, dataframe.iloc[i].values.reshape(1,-1)), i))
        
        except IndexError:
            continue
    
    # Sort cosine similarity recs and select the 500 closest, excluding the target
    cos_sim = [i[1] for i in sorted(cos_sim[:-2:])][0:500]
    
    # Sort color distance recs and select 500, excluding the target
    color_output = [i[1] for i in sorted(color_output)][1:501]
    
    # Add model weights to the predictions if selected- doubling the frequency of
    # the top 50 recommendations from the color recommender or cosine similarity 
    # function
    if model_weights == 'visual':
        e = cos_sim + color_output + color_output[:50] + knn + kmean
    elif model_weights == 'cos_sim':
        e = cos_sim + cos_sim[:50] + color_output + knn + kmean
    else:
        e = cos_sim + color_output + knn + kmean
    
    # Count the instances of each index being recommended
    rec_dict = Counter(e)
    
    # Separate keys and values into lists
    f = list(rec_dict.keys())
    g = list(rec_dict.values())
    
    # Select the index of the most occuring values
    max_ = [i for i, x in enumerate(g) if x == max(g)]
    
    # Add the keys for each of the most occuring values by index
    recs = [f[i] for i in max_]
    
    # If there are 10 recommendations in the list, return, 
    # otherwise continue the process for the next lowest number of recs
    if len(recs) >= 10:
        return recs[:10]
    max_minus = [i for i, x in enumerate(g) if x == max(g)-1]
    recs = recs + [f[i] for i in max_minus]
    if len(recs) >= 10:
        return recs[:10]
    max_minus_2 = [i for i, x in enumerate(g) if x == max(g)-2]
    recs = recs + [f[i] for i in max_minus_2]

    return recs[:10]

#=============================================== Evaluation Metrics ======================================================
 
def evaluate_recs(target_index, recommendation_indices, dataframe, colordata):
    '''Evaluate the quality of recommendations based on four metrics- Cosine Similarity, 
       color distance, standard deviation of likes and difference in mean price from target.
       
       Requires 4 arguments- the target index, recommendation indices, the dataframe and
       the dataframe of color tuples.'''
    
    list_of_columns = [colordata['0'], colordata['1'], colordata['2']]
    distance = []
    cos_sim = []
    # Iterate through indices and calculate Euclidean Distance and cosine similarity
    for i in recommendation_indices[1:]:
        cos_sim.append(cosine_similarity(dataframe.loc[[target_index]], dataframe.loc[[i]]))
        distance.append(np.mean(
            [dist.euclidean(list_of_columns[0][target_index], list_of_columns[0][i]),
            dist.euclidean(list_of_columns[1][target_index], list_of_columns[1][i]),
            dist.euclidean(list_of_columns[2][target_index], list_of_columns[2][i])]))
    
    # Print standard deviations and mean price difference
    print('Average Cosine Similarity with target item:', np.mean(cos_sim))
    print('Average Euclidean Color Distance from target item:', np.mean(distance))  
    print('Standard Deviation of likes in Recommendation Group:', dataframe['FollowerCount'][recommendation_indices].std())
    print('Difference in mean price and target item:', (dataframe.Price[recommendation_indices].mean()- dataframe.Price[target_index]))

def show_recs(target, recs):
    '''Takes a list of recommendation indices and a target and displays them all'''
    images = []
    images.append(Image.open(f'./src/images/{target}.jpg'))
    for i in recs:
        images.append(Image.open(f'./src/images/{i}.jpg'))
    
    # Display the original image and top five closest
    fig, ax = plt.subplots(2, 5, figsize = (12,6))
    fig.suptitle('Ensemble Recommendations', size = 16)
    ax[0,0].imshow(images[0])
    ax[0,0].axis('off')
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(images[1])
    ax[0,1].axis('off')
    ax[0,1].set_title('Recommendation 1')
    ax[0,2].imshow(images[2])
    ax[0,2].axis('off')
    ax[0,2].set_title('Recommendation 2')
    ax[0,3].imshow(images[3])
    ax[0,3].axis('off')
    ax[0,3].set_title('Recommendation 3')
    ax[0,4].imshow(images[4])
    ax[0,4].axis('off')
    ax[0,4].set_title('Recommendation 4')
    ax[1,0].imshow(images[5])
    ax[1,0].axis('off')
    ax[1,0].set_title('Recommendation 5')
    ax[1,1].imshow(images[6])
    ax[1,1].axis('off')
    ax[1,1].set_title('Recommendation 6')
    ax[1,2].imshow(images[7])
    ax[1,2].axis('off')
    ax[1,2].set_title('Recommendation 7')
    ax[1,3].imshow(images[8])
    ax[1,3].axis('off')
    ax[1,3].set_title('Recommendation 8')
    ax[1,4].imshow(images[9])
    ax[1,4].axis('off')
    ax[1,4].set_title('Recommendation 9')

    return plt.show()

def evaluate_model(dataframe, color_dataframe, model_weights = None, n_samples = 20):
    '''Runs the ensemble model n number of times with randomly selected target
       rows and returns average cosine similarity, color distance, standard deviation
       of likes and mean price difference from each of the recommendations.'''
    
    # Create an empty dictionary for average metrics
    evaluations = {}
    
    # List of columns for color distance
    list_of_columns = [color_dataframe['0'], color_dataframe['1'], color_dataframe['2']]
    
    # Empty lists for each model iteration's evaluation metrics
    avg_cos_sim = []
    avg_dist = []
    avg_std_likes = []
    avg_price_dif = []

    # Iterate through n random rows as targets
    for i in np.random.choice(dataframe.index, size = n_samples, replace= False):
        
        # Create recommendations for each row
        recs = ensemble_recs(dataframe.loc[[i]], dataframe, color_dataframe, model_weights= model_weights)
        
        # Empty lists for individual item metrics
        cos_sim = []
        distance = []

        # Iterate through recommendations and evaluate individually, append averages to iteration lists
        for j in recs:
            cos_sim.append(cosine_similarity(dataframe.loc[[i]], dataframe.loc[[j]]))
            distance.append(np.mean(
               [dist.euclidean(list_of_columns[0][i], list_of_columns[0][j]),
                dist.euclidean(list_of_columns[1][i], list_of_columns[1][j]),
                dist.euclidean(list_of_columns[2][i], list_of_columns[2][j])]))
        avg_cos_sim.append(np.mean(cos_sim))
        avg_dist.append(np.mean(distance))
        avg_std_likes.append(dataframe['FollowerCount'][recs].std())
        avg_price_dif.append(dataframe.Price[recs].mean() - dataframe.Price[i])
    
    # Update dictionary with averages of every iteration's performance
    evaluations.update({'cos_sim' : np.mean(avg_cos_sim),
                        'distance' : np.mean(avg_dist),
                        'std_likes' : np.mean(avg_std_likes),
                        'mean_price_dif' : np.mean(avg_price_dif)})
    
    # Return evaluation averages
    return evaluations