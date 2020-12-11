import numpy as np
import cv2 as cv
from colorthief import ColorThief
import os

def plot_color_palette(img_path):
    image_vec = cv.imread(img_path, 1)
    RGB_img = cv.cvtColor(image_vec, cv.COLOR_BGR2RGB)

    color_thief = ColorThief(img_path)
    color_palette = color_thief.get_palette(quality = 1, color_count=3)

    palette = np.array(color_palette)[np.newaxis, :, :]

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,6))

    ax2.imshow(palette);
    ax2.axis('off');
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

    for i, filename in enumerate(os.listdir(source_directory)):
        if filename.endswith(".jpg"):
            image = cv.imread(f'{source_directory}{filename}', 1)
            RGB_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            new_img = remove_background(RGB_img)
            cv.imwrite(f'{destination_directory}{filename}', new_img)
            if i%50 == 0:
                print(f'{i} Images Converted')
    return print('All Images Converted')