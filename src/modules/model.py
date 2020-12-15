import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import scipy.spatial.distance as dist

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
    images = []
    images.append(Image.open(f'./src/images/{input_row.index[0]}.jpg'))
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))
    
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