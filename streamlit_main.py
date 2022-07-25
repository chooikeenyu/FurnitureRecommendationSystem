import cv2
import pandas as pd
from keras.models import load_model

import streamlit as st
import os
from PIL import Image, ImageOps
import tensorflow
import numpy as np
import pickle
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from IPython.display import display
from st_clickable_images import clickable_images

st.title('Furniture Recommendation System')

feature_list_Bathroom = np.array(pickle.load(open('embeddings_Bathroom.pkl','rb')))
feature_list_Bedroom = np.array(pickle.load(open('embeddings_Bedroom.pkl','rb')))
feature_list_Dining = np.array(pickle.load(open('embeddings_Dining.pkl','rb')))
feature_list_Game = np.array(pickle.load(open('embeddings_Game.pkl','rb')))
feature_list_HomeOffice = np.array(pickle.load(open('embeddings_HomeOffice.pkl','rb')))
feature_list_Kids = np.array(pickle.load(open('embeddings_Kids.pkl','rb')))
feature_list_Kitchen = np.array(pickle.load(open('embeddings_Kitchen.pkl','rb')))
feature_list_Living = np.array(pickle.load(open('embeddings_Living.pkl','rb')))
feature_list_Replacement = np.array(pickle.load(open('embeddings_Replacement.pkl','rb')))

filenames = pickle.load(open('filenames.pkl','rb'))
filenames_Bath = pickle.load(open('filenames_Bath.pkl','rb'))
filenames_Bed = pickle.load(open('filenames_Bed.pkl','rb'))
filenames_Dining = pickle.load(open('filenames_Dining.pkl','rb'))
filenames_Game = pickle.load(open('filenames_Game.pkl','rb'))
filenames_Home = pickle.load(open('filenames_Home.pkl','rb'))
filenames_Kids = pickle.load(open('filenames_Kids.pkl','rb'))
filenames_kitchen = pickle.load(open('filenames_Kitchen.pkl','rb'))
filenames_Living = pickle.load(open('filenames_Living.pkl','rb'))
filenames_Replace = pickle.load(open('filenames_Replace.pkl','rb'))
model = load_model("imgaug_rmspropV2_Ortho_2.h5")


def save_upload_file(upload_file):
    try:
        with open(os.path.join('uploads',upload_file.name),'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    #img = image.load_img(img_path, target_size=(224, 224)) #local database
    img = ImageOps.fit(img_path,(224,224), Image.ANTIALIAS)
    img_array = np.asarray(img)
    #img_array = image.img_to_array(img) #local database
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    preprocessed_img = preprocessed_img/255
    # preprocess the image by substract the mean value from each channel of images in the batch
    # mean is the array of elements obtained by average RGB pixels of all images obtained from imagenet
    result = model.predict(preprocessed_img).flatten()
    # result1 = model.predict(preprocessed_img)
    # pred_class = decode_predictions(result1,top=1)[0]
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    return indices


def readDataFrame(file_list,x):
    df = pd.DataFrame() # empty dataframe
    #path = "dataset"
    # for element in file_list:
    #     element = os.path.join(path,element)
    #     element = element.replace(os.sep,'/')
    df = pd.read_excel(file_list)
    prod_info = df.loc[df['image_id'] == x]
    return prod_info


upload_file = st.file_uploader("First step: Select a Furniture Image to upload",type=['png','jpeg','jpg'])
img_file = []

if upload_file is not None:
    #display_image = Image.open(upload_file)
    #st.image(display_image)

    #save_upload_file(upload_file) #local database
    #features = feature_extraction(os.path.join("uploads", upload_file.name), model) #local database
    img = Image.open(upload_file)
    features = feature_extraction(img,model)

    maxindex = features.argmax()
    if maxindex == 0:
        indices = recommend(features, feature_list_Bathroom)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Bath[indices[0][x]])
    elif maxindex == 1:
        indices = recommend(features, feature_list_Bedroom)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Bed[indices[0][x]])
    elif maxindex == 2:
        indices = recommend(features, feature_list_Dining)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Dining[indices[0][x]])
    elif maxindex == 3:
        indices = recommend(features, feature_list_Game)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Game[indices[0][x]])
    elif maxindex == 4:
        indices = recommend(features, feature_list_HomeOffice)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Home[indices[0][x]])
    elif maxindex == 5:
        indices = recommend(features, feature_list_Kids)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Kids[indices[0][x]])
    elif maxindex == 6:
        indices = recommend(features, feature_list_Kitchen)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_kitchen[indices[0][x]])
    elif maxindex == 7:
        indices = recommend(features, feature_list_Living)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Living[indices[0][x]])
    elif maxindex == 8:
        indices = recommend(features, feature_list_Replacement)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Replace[indices[0][x]])


img_disp = []
df_list = []
for x in img_file:
    if x[0] == "0":
        img_dir = 'dataset/cleaned_Bathroom Furniture.xlsx'
        new_df = readDataFrame(img_dir,x)
        img_dir = 'images3/Bathroom/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)
        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "1":
        img_dir = 'dataset/cleaned_Bedroom Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Bedroom/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "2":
        img_dir = 'dataset/cleaned_Diningroom Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Diningroom/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "3":
        img_dir = 'dataset/cleaned_Gameroom Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Gameroom/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "4":
        img_dir = 'dataset/cleaned_HomeOffice Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/HomeOffice/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "5":
        img_dir = 'dataset/cleaned_Kids Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Kids/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "6":
        img_dir = 'dataset/cleaned_Kitchen Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Kitchen/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "7":
        img_dir = 'dataset/cleaned_Livingroom Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/Livingroom/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)


    elif x[0] == "8":
        img_dir = 'dataset/cleaned_Replacementparts Furniture.xlsx'
        new_df = readDataFrame(img_dir, x)
        img_dir = 'images3/ReplacementParts/'
        img_path = '{}{}'.format(img_dir, x)

        img = cv2.imread(img_path)

        img_disp.append(img)
        df_list.append(new_df)

if upload_file is not None:
    choices = st.radio("Second step: Choose top k recommendation",("TOP 5","TOP 8","TOP 10"))

    button = st.button("Click here to get recommendations!")

    if button:
        st.success('Hope you like the below recommendations :)')
        if choices == 'TOP 5':
            recomm = []
            for x in range(5):
                recomm.append("Recommendation " + str(x+1))

            if bool(img_disp):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(recomm)
                with tab1:
                    st.header(df_list[0].loc[:,"Title"].item())
                    st.image(img_disp[0], width=250)
                    #st.write(df_list[0])
                    # st.markdown("""
                    # <style>
                    # .big-font {
                    #     font-size:30px;
                    # }
                    # </style>
                    # """, unsafe_allow_html=True)
                    # URL = '<p class="big-font">URL: </p>' + str(df_list[0].loc[:,"Url"].item())
                    # st.markdown(URL , unsafe_allow_html=True)
                    st.write("URL: " + df_list[0].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[0].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[0].loc[:,"Price"].item())
                with tab2:
                    st.header(df_list[1].loc[:,"Title"].item())
                    st.image(img_disp[1], width=250)
                    #st.write(df_list[1])
                    st.write("URL: " + df_list[1].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[1].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[1].loc[:,"Price"].item())
                with tab3:
                    st.header(df_list[2].loc[:,"Title"].item())
                    st.image(img_disp[2], width=250)
                    #st.write(df_list[2])
                    st.write("URL: " + df_list[2].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[2].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[2].loc[:,"Price"].item())
                with tab4:
                    st.header(df_list[3].loc[:,"Title"].item())
                    st.image(img_disp[3], width=250)
                    #st.write(df_list[3])
                    st.write("URL: " + df_list[3].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[3].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[3].loc[:,"Price"].item())
                with tab5:
                    st.header(df_list[4].loc[:,"Title"].item())
                    st.image(img_disp[4], width=250)
                    #st.write(df_list[4])
                    st.write("URL: " + df_list[4].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[4].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[4].loc[:,"Price"].item())

        elif choices == 'TOP 8':
            recomm = []
            for x in range(8):
                recomm.append("Recommendation " + str(x+1))

            if bool(img_disp):
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(recomm)
                with tab1:
                    st.header(df_list[0].loc[:,"Title"].item())
                    st.image(img_disp[0], width=250)
                    #st.write(df_list[0])
                    # st.markdown("""
                    # <style>
                    # .big-font {
                    #     font-size:30px;
                    # }
                    # </style>
                    # """, unsafe_allow_html=True)
                    # URL = '<p class="big-font">URL: </p>' + str(df_list[0].loc[:,"Url"].item())
                    # st.markdown(URL , unsafe_allow_html=True)
                    st.write("URL: " + df_list[0].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[0].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[0].loc[:,"Price"].item())
                with tab2:
                    st.header(df_list[1].loc[:,"Title"].item())
                    st.image(img_disp[1], width=250)
                    #st.write(df_list[1])
                    st.write("URL: " + df_list[1].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[1].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[1].loc[:,"Price"].item())
                with tab3:
                    st.header(df_list[2].loc[:,"Title"].item())
                    st.image(img_disp[2], width=250)
                    #st.write(df_list[2])
                    st.write("URL: " + df_list[2].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[2].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[2].loc[:,"Price"].item())
                with tab4:
                    st.header(df_list[3].loc[:,"Title"].item())
                    st.image(img_disp[3], width=250)
                    #st.write(df_list[3])
                    st.write("URL: " + df_list[3].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[3].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[3].loc[:,"Price"].item())
                with tab5:
                    st.header(df_list[4].loc[:,"Title"].item())
                    st.image(img_disp[4], width=250)
                    #st.write(df_list[4])
                    st.write("URL: " + df_list[4].loc[:,"Url"].item())
                    st.write("Average rating: " + df_list[4].loc[:,"Avg_ratings"].item())
                    st.write("Price: " + df_list[4].loc[:,"Price"].item())
                with tab6:
                    st.header(df_list[5].loc[:, "Title"].item())
                    st.image(img_disp[5], width=250)
                    # st.write(df_list[4])
                    st.write("URL: " + df_list[5].loc[:, "Url"].item())
                    st.write("Average rating: " + df_list[5].loc[:, "Avg_ratings"].item())
                    st.write("Price: " + df_list[5].loc[:, "Price"].item())
                with tab7:
                    st.header(df_list[6].loc[:, "Title"].item())
                    st.image(img_disp[6], width=250)
                    # st.write(df_list[4])
                    st.write("URL: " + df_list[6].loc[:, "Url"].item())
                    st.write("Average rating: " + df_list[6].loc[:, "Avg_ratings"].item())
                    st.write("Price: " + df_list[6].loc[:, "Price"].item())
                with tab8:
                    st.header(df_list[7].loc[:, "Title"].item())
                    st.image(img_disp[7], width=250)
                    # st.write(df_list[4])
                    st.write("URL: " + df_list[7].loc[:, "Url"].item())
                    st.write("Average rating: " + df_list[7].loc[:, "Avg_ratings"].item())
                    st.write("Price: " + df_list[7].loc[:, "Price"].item())