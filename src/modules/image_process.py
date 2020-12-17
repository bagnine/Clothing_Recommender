import numpy as np
import cv2 as cv
from colorthief import ColorThief
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_color_palette(img_path):
    '''Takes in an image path, pulls the color palette from that image 
       using ColorThief and displays the image and color palette side by side'''
    
    # Read in the image, convert to RGB
    image_vec = cv.imread(img_path, 1)
    RGB_img = cv.cvtColor(image_vec, cv.COLOR_BGR2RGB)

    # Instantiate ColorThief and pull palette
    color_thief = ColorThief(img_path)
    color_palette = color_thief.get_palette(quality = 1, color_count=3)

    # Reshape color array and plot
    palette = np.array(color_palette)[np.newaxis, :, :]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,6))

    ax2.imshow(palette)
    ax2.axis('off')
    ax1.imshow(RGB_img)
    ax1.set_title('Original Image')
    plt.show();

def remove_background(rgb_img):
    '''This code block is adapted from Chris Albon
       https://chrisalbon.com/machine_learning/preprocessing_images/remove_backgrounds/
       Given an RBG image ~700x700, creates a rectangle in the center of the frame, 
       analyzes the area outside the frame, and using cv2 GrabCut, subtracts everything 
       inside that looks like the outside '''
    
    # Create initial mask
    mask = np.zeros(rgb_img.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Creating a rectangle to select the center of the image
    rectangle = (80, 40, 600, 600)

    # Run grabCut
    cv.grabCut(rgb_img, # Our image
            mask, # The Mask
            rectangle, # Our rectangle
            bgdModel, # Temporary array for background
            fgdModel, # Temporary array for background
            5, # Number of iterations
            cv.GC_INIT_WITH_RECT) # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    image_rgb_nobg = rgb_img * mask_2[:, :, np.newaxis]
    return image_rgb_nobg

def convert_images(source_directory, destination_directory):
    '''Takes in a source and destination directory, loads the images in the source,
       removes their background and saves them as the same file name in the destination
       directory'''
    
    # Iterate through all files in the directory
    for i, filename in enumerate(os.listdir(source_directory)):
        if filename.endswith(".jpg"):
            image = cv.imread(f'{source_directory}{filename}', 1)
            RGB_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            new_img = remove_background(RGB_img)
            cv.imwrite(f'{destination_directory}{filename}', new_img)
            if i%50 == 0:
                print(f'{i} Images Converted')
    return print('All Images Converted')

def color_palette_dataframe(filepath):
    '''Pulls the 3 most dominant colors in an image palette and returns of DataFrame
       with the filename as index and the RGB values as columns'''
    # Create an empty dictionary
    color_dict = {}

    # Iterate through all files in the chosen directory
    for i, filename in enumerate(os.listdir(filepath)):
        
        # Filter out any non-image files
        if filename.endswith('.jpg'):
            
            # Instantiate ColorThief and get the color palette
            color_thief = ColorThief(f'{filepath}{filename}')
            color_palette = color_thief.get_palette(quality = 1, color_count=3)
            
            # Create a new list of the sums of each RGB tuple
            new_list = [sum(i) for i in color_palette]
            
            # Remove the lowest RGB value from the list- this will be the black in the background of each image
            # and add the remaining values to the dictionary with the image number as index
            color_dict.update({filename.strip('.jpg') : [i for i in color_palette if sum(i) != min(new_list)]})
            
            # Progress counter
            if i%100 == 0:
                print(f'{i} files complete')
    
    # Convert the dictionary to a dataframe with the filename as index
    return pd.DataFrame.from_dict(color_dict, orient = 'index')

def convert_directory_to_rgb(source_file_path, dest_file_path):
    '''Takes a directory of images in BGR format and converts and saves them as RGB'''
    
    # Iterate through all files in the chosen directory
    for filename in os.listdir(source_file_path):
        
        # Choose only image files
        if filename.endswith('.jpg'):
            
            # Load images, convert them to RGB and save
            img = cv.imread(f'{source_file_path}{filename}')
            RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            cv.imwrite(f'{dest_file_path}{filename}', RGB_img)
    
    #Print statement on completion
    return print('Color correction complete')