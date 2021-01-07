import streamlit as st
from PIL import Image
import os
import image_process
import model
from colorthief import ColorThief
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('../data/model_data.csv')
df = df[['Unnamed: 0','Price', 'VINTAGE', 'XS', 'S', 'M', 'L', 'XL', 'XXL']]
df = df.set_index('Unnamed: 0')
coldf = pd.read_csv('../data/colors.csv')
coldf = coldf.set_index('Unnamed: 0')

coldf['0'] = coldf['0'].apply(lambda x: eval(x))
coldf['1'] = coldf['1'].apply(lambda x: eval(x))
coldf['2'] = coldf['2'].apply(lambda x: eval(x))

c1_tuple, c2_tuple, c3_tuple = (0,0,0), (0,0,0), (0,0,0)

def show_recs(recs, target = None):
    '''Takes a list of recommendation indices and a target and displays them all'''
    images = []
    sizes = []
    prices = []
    if target != None:
        images.append(Image.open(f'/Users/nicksubic/Documents/flatiron/phase_1/nyc-mhtn-ds-091420-lectures/capstone/Clothing_Recommender/src/images/{target}.jpg'))
    for i in recs:
        images.append(Image.open(f'/Users/nicksubic/Documents/flatiron/phase_1/nyc-mhtn-ds-091420-lectures/capstone/Clothing_Recommender/src/images/{i}.jpg'))
        size_list = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        for j in size_list:
            if int(df.loc[[i]][j]) == 1:
                sizes.append(j)
        prices.append(int(df.loc[[i]]['Price']))
    
    # Display the original image and top five closest
    fig, ax = plt.subplots(2, 5, figsize = (12,6))
    fig.suptitle('Recommendations', size = 16)
    ax[0,0].imshow(images[0])
    ax[0,0].axis('off')
    ax[0,0].set_title(f'Size: {sizes[0]}  Price: ${prices[0]}')
    ax[0,1].imshow(images[1])
    ax[0,1].axis('off')
    ax[0,1].set_title(f'Size: {sizes[1]}  Price: ${prices[1]}')
    ax[0,2].imshow(images[2])
    ax[0,2].axis('off')
    ax[0,2].set_title(f'Size: {sizes[2]}  Price: ${prices[2]}')
    ax[0,3].imshow(images[3])
    ax[0,3].axis('off')
    ax[0,3].set_title(f'Size: {sizes[3]}  Price: ${prices[3]}')
    ax[0,4].imshow(images[4])
    ax[0,4].axis('off')
    ax[0,4].set_title(f'Size: {sizes[4]}  Price: ${prices[4]}')
    ax[1,0].imshow(images[5])
    ax[1,0].axis('off')
    ax[1,0].set_title(f'Size: {sizes[5]}  Price: ${prices[5]}')
    ax[1,1].imshow(images[6])
    ax[1,1].axis('off')
    ax[1,1].set_title(f'Size: {sizes[6]}  Price: ${prices[6]}')
    ax[1,2].imshow(images[7])
    ax[1,2].axis('off')
    ax[1,2].set_title(f'Size: {sizes[7]}  Price: ${prices[7]}')
    ax[1,3].imshow(images[8])
    ax[1,3].axis('off')
    ax[1,3].set_title(f'Size: {sizes[8]}  Price: ${prices[8]}')
    ax[1,4].imshow(images[9])
    ax[1,4].axis('off')
    ax[1,4].set_title(f'Size: {sizes[9]}  Price: ${prices[9]}')

    return plt.show()

st.markdown('### Visual Apparel Recommender')

size_list = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
size = st.multiselect('Select your size:', (size_list))
vintage = st.selectbox('New or vintage:', ('New', 'Vintage'))
price = st.number_input('Select your ideal price:', min_value = None, max_value = None)
upload = st.radio('Search by color or by image:', ('Color', 'Image'))
if upload == 'Image':
    st.write('Upload image here:')
    if st.button('Select Image'):
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
        image_ = Image.open(uploaded_file)
        st.image(image_, caption='Uploaded Image', use_column_width=True)
        if st.button('Recommend!') and uploaded_file is not None:
            
            Image.imshow
    #         vector = image_process.remove_background(image_)
    #         ct = ColorThief(vector)
    #         palette = ct.get_palette(quality = 1, color_count = 3)
    #         new_list = [sum(i) for i in palette]
    #         selected_palette = [i for i in palette if sum(i) != min(new_list)]
    #         st.write(selected_palette)
    #         c1_tuple, c2_tuple, c3_tuple = selected_palette[0], selected_palette[1], selected_palette[2]
    #         if st.button('Recommend! '):
    #             st.write(c1_tuple)
    #     elif st.button('Recommend!  ') and image_ == None:
    #         st.warning('You must upload a file or select colors')
    #     else:
    #         st.stop()
    # else:
    #     st.stop()

else:
    st.write('Select 3 colors:')
    color1 = st.color_picker('Choose color 1')
    color2 = st.color_picker('Choose color 2')
    color3 = st.color_picker('Choose color 3')
    
    c1_tuple = (tuple(int(color1[i:i+2], 16) for i in (1, 3, 5)))
    c2_tuple = (tuple(int(color2[i:i+2], 16) for i in (1, 3, 5)))
    c3_tuple = (tuple(int(color3[i:i+2], 16) for i in (1, 3, 5)))

if size != None and price != 0.00 and c1_tuple != (0, 0, 0) and c2_tuple != (0,0,0) and c3_tuple != (0,0,0):
    if st.button('Recommend!'):
        if vintage =='New':
            vintage = 0
        else:
            vintage = 1
        size_val = [1 if i in size else 0 for i in size_list]
        columns = ['Price', 'VINTAGE']+ size_list
        columns1 = ['0', '1', '2']

        d1 = pd.DataFrame.from_dict({110000: [price, vintage]+ size_val }, columns = columns, orient = 'index')
        d2 = pd.DataFrame.from_dict({110000: [c1_tuple, c2_tuple, c3_tuple]}, columns = columns1, orient = 'index')

        df = pd.concat([df, d1], axis = 0)
        coldf = pd.concat([coldf, d2], axis = 0)

        recs = model.ensemble_recs(df.iloc[[-1]], df, coldf, model_weights = 'visual')
        fig = show_recs(recs)
        st.pyplot(fig)
