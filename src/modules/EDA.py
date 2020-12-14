import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import scipy.spatial.distance as dist
from PIL import Image

#================================================= Data Cleaning ============================================

def dollars_to_int(column):
    ''''''
    column = column.str.replace('$', '')
    column = column.str.replace(',', '')
    return pd.to_numeric(column) 

def format_listings(column):
    ''''''
    listings = []
    for i in column:
        try:
            listings.append(int(i.split()[0]))
        except AttributeError:
            listings.append(i)
    return listings

def bin_designers(column):
    ''''''
    for i in [0,1]:
        for i, name in enumerate(column):
            name = str(name)
            name.strip()
            if 'NIKE' in name:
                column[i] = 'NIKE'
            elif 'ADIDAS' in name:
                column[i] = 'ADIDAS'
            elif 'COMME DES GARCONS' in name:
                column[i] = 'COMME DES GARCONS'
            elif 'SUPREME' in name:
                column[i] = 'SUPREME'
            elif 'PALACE' in name:
                column[i] = 'PALACE'
            elif 'CHROME' in name:
                column[i] = 'CHROME HEARTS'
            elif 'FEAR' in name:
                column[i] = 'FEAR OF GOD'
            elif 'WHITE' in name:
                column[i] = 'OFF WHITE'
            elif 'KAPITAL' in name:
                column[i] = 'KAPITAL'
            elif 'BAPE' in name:
                column[i] = 'BAPE'
            elif 'KANYE' in name or 'YEEZY' in name:
                column[i] = 'YEEZY SEASON'
            elif 'LAUREN' in name:
                column[i] = 'RALPH LAUREN'
            elif 'ISSEY' in name or 'YAMAMOTO' in name or 'JAPANESE DESIGNER' in name:
                column[i] = 'JAPANESE DESIGNER'
            elif 'LAURENT' in name:
                column[i] = 'SAINT LAURENT'
            elif 'ACNE' in name:
                column[i] = 'ACNE STUDIOS'
            elif 'RAF' in name:
                column[i] = 'RAF SIMMONS'
            elif 'VETEMENTS' in name:
                column[i] = 'VETEMENTS'
            elif 'GIVENCHY' in name:
                column[i] = 'GIVENCHY'
            elif 'OWENS' in name:
                column[i] = 'RICK OWENS'
            elif 'GUCCI' in name:
                column[i] = 'GUCCI'
            elif 'PRADA' in name:
                column[i] = 'PRADA'
            elif 'MARGIELA' in name:
                column[i] = 'MAISON MARGIELA'
            elif 'UNIQLO' in name:
                column[i] = 'UNIQLO'
            elif 'SOCIAL' in name:
                column[i] = 'ANTI SOCIAL SOCIAL CLUB'
            elif 'STUSSY' in name:
                column[i] = 'STUSSY'
            elif 'DISNEY' in name:
                column[i] = 'DISNEY'
            elif 'CHAMPION' in name:
                column[i] = 'CHAMPION'
            elif 'HAWAIIAN' in name or 'CRAZY SHIRTS' in name or 'SURF' in name:
                column[i] = 'HAWAIIAN SHIRT'
            elif 'BAND TEES' in name:
                column[i] = 'BAND TEES'
            elif 'JAPANESE BRAND' in name:
                column[i] = 'JAPANESE BRAND'
            elif 'STREETWEAR' in name:
                column[i] = 'STREETWEAR'
            elif 'VINTAGE' in name:
                column[i] = 'VINTAGE'
            else:
                column[i] = 'MISC'

    return column

#================================================= Filtering Images by Color =============================================

def find_closest_colors(list_of_tuples, list_of_columns, dataframe):
    ''''''
    # Creating an empty list for the distances that will be caculated
    output = []
    
    # Iterating through the dataframe and comparing each color to those of my input image with
    # RGB values expressed as 3d coordinates, then taking the sum of all distances and saving as
    # a tuple with each index number
    for i in dataframe.index:
        output.append((sum(
            [dist.euclidean(list_of_tuples[0], list_of_columns[0][i]),
            dist.euclidean(list_of_tuples[1], list_of_columns[1][i]),
            dist.euclidean(list_of_tuples[2], list_of_columns[2][i])]), i))
    
    # Taking the indices of the 5 closest items, plus the original (it will always be first because
    # its distance to itself is 0)
    indices = [i[1] for i in sorted(output)[0:6]]
    
    # Returning the DataFrame rows of the nearest color palettes
    return dataframe.iloc[indices]


def display_closest_colors(item_index, list_of_columns, dataframe):
    ''''''
    # Get RGB data from the target item
    tuples = [list_of_columns[0][item_index],
              list_of_columns[1][item_index],
              list_of_columns[2][item_index]]
    
    # Create and empty list for euclidean distance
    output = []
    
    # Iterate through the dataframe and find euclidean distance between each of the colors of
    # each item in the dataframe and the target, then calculate the total distance between all
    # three colors. Save to the list as a tuple- the first value is the sum Euclidean distance,
    # the second is the index of the item
    for i in dataframe.index:
        output.append((sum(
            [dist.euclidean(tuples[0], list_of_columns[0][i]),
            dist.euclidean(tuples[1], list_of_columns[1][i]),
            dist.euclidean(tuples[2], list_of_columns[2][i])]), i))
    
    # Sort the euclidean distances and pull the 6 closest entries (the first one will always be
    # target item as its distance to itself is 0)
    indices = [i[1] for i in sorted(output)[0:7]]
    images = []
    
    # Open the images from each of the closest images
    for i in indices:
        images.append(Image.open(f'./src/images/{i}.jpg'))

    # Display the original image and top five closest
    fig, ax = plt.subplots(2, 3, figsize = (10,6))
    fig.suptitle('Similar items based on color palette', size = 16)
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