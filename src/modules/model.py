import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import scipy.spatial.distance as dist
from sklearn.neighbors import NearestNeighbors

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
    ''''''
    
    # Take indices from clusters with the target row
    indices = cluster_recommendations(input_row, dataframe, n_clusters)
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    images = []
    images.append(Image.open(f'./src/images/{input_row.index[0]}.jpg'))
    test_i = []
    for i in np.random.choice(indices, size=6, replace=True):
        images.append(Image.open(f'./src/images/{i}.jpg'))
        test_i.append(i)
    for i in test_i:
        print(f'Image {i} cosine similarity:', cosine_similarity(input_row, dataframe.loc[i:i]))
              
    print('Standard Deviation of likes in Recommendation Group:', dataframe['FollowerCount'][test_i].std())
    print('Difference in mean price and target item:', (dataframe.Price[test_i].mean()))
              
    # Display the images
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on cosine similarity', size = 16)
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

def KNN_recs(input_row, dataframe, n_neighbors):
    knn = NearestNeighbors(n_neighbors= n_neighbors, n_jobs=-1)
    knn.fit(dataframe)

    recs = knn.kneighbors(X= input_row, n_neighbors=500, return_distance=False)
    return recs

def display_KNN_recs(input_row, dataframe, n_neighbors):
    
    # Take indices from nearest neighbors
    indices = KNN_recs(input_row, dataframe, n_neighbors)[0]
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    images = []
    test_i = []
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))
        test_i.append(i)
    for i in test_i:
        print(f'Image {i} cosine similarity:', cosine_similarity(input_row, dataframe.loc[i:i]))
              
    print('Standard Deviation of likes in Recommendation Group:', dataframe['FollowerCount'][test_i].std())
    print('Difference in mean price and target item:', (input_row.Price - dataframe.Price[test_i].mean()))
              
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
    ''''''
    
    # Take indices from closest cosine similarities to the target row
    indices = [i[1] for i in cosine_similarity_recs(input_row, dataframe)]
    
    # Create an empty list and append the images from each indice to it, starting
    # with the input row
    test_i = []
    images = []
    images.append(Image.open(f'./src/images/{input_row.index[0]}.jpg'))
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))
        test_i.append(i)
    for i in test_i:
        print(f'Image {i} cosine similarity:', cosine_similarity(input_row, dataframe.loc[i:i]))
              
    print('Standard Deviation of likes in Recommendation Group:', dataframe['FollowerCount'][test_i].std())
    print('Difference in mean price and target item:', (input_row.Price - dataframe.Price[test_i].mean()))
    
    # Display the images
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on cosine similarity', size = 16)
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