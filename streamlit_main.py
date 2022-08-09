import cv2
import pandas as pd
from keras.models import load_model

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pickle
from keras.applications.resnet import preprocess_input
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt

import seaborn as sns
import re

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


def feature_extraction(img_path,model):
    #img = image.load_img(img_path, target_size=(224, 224)) #local database
    img = ImageOps.fit(img_path,(224,224), Image.ANTIALIAS) # cloud
    img_array = np.asarray(img) # cloud
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
    neighbors = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])

    return indices,distances


def readDataFrame(file_list,x):
    df = pd.DataFrame() # empty dataframe
    #path = "dataset"
    # for element in file_list:
    #     element = os.path.join(path,element)
    #     element = element.replace(os.sep,'/')
    df = pd.read_excel(file_list)
    prod_info = df.loc[df['image_id'] == x]
    return prod_info


def append_imgfilename(maxindex,title_distance = None):
    global img_file
    if title_distance is not None:
        img_file = []
    if maxindex == 0:
        indices, distances = recommend(features, feature_list_Bathroom)
        if title_distance is not None:
            x3 = np.add(title_distance, distances) #distances

            indices = indices.flatten()  # indices
            x3 = x3.flatten()  # distances
            sorter = np.argsort(x3)
            x3 = x3[sorter]
            x3= x3.reshape(1, -1)
            indices = indices[sorter]
            indices = indices.reshape(1, -1)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Bath[indices[0][x]])
    elif maxindex == 1:
        indices, distances = recommend(features, feature_list_Bedroom)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Bed[indices[0][x]])
    elif maxindex == 2:
        indices, distances = recommend(features, feature_list_Dining)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Dining[indices[0][x]])
    elif maxindex == 3:
        indices, distances = recommend(features, feature_list_Game)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Game[indices[0][x]])
    elif maxindex == 4:
        indices, distances = recommend(features, feature_list_HomeOffice)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Home[indices[0][x]])
    elif maxindex == 5:
        indices, distances = recommend(features, feature_list_Kids)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Kids[indices[0][x]])
    elif maxindex == 6:
        indices, distances = recommend(features, feature_list_Kitchen)
        if title_distance is not None:
            x3 = np.add(title_distance, distances) #distances

            indices = indices.flatten()  # indices
            x3 = x3.flatten()  # distances
            sorter = np.argsort(x3)
            x3 = x3[sorter]
            x3= x3.reshape(1, -1)
            indices = indices[sorter]
            indices = indices.reshape(1, -1)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_kitchen[indices[0][x]])
    elif maxindex == 7:
        indices, distances = recommend(features, feature_list_Living)
        if title_distance is not None:
            x3 = np.add(title_distance, distances) #distances

            indices = indices.flatten()  # indices
            x3 = x3.flatten()  # distances
            sorter = np.argsort(x3)
            x3 = x3[sorter]
            x3= x3.reshape(1, -1)
            indices = indices[sorter]
            indices = indices.reshape(1, -1)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Living[indices[0][x]])
    elif maxindex == 8:
        indices, distances = recommend(features, feature_list_Replacement)
        a = len(indices[0])
        for x in range(a):
            img_file.append(filenames_Replace[indices[0][x]])

    return img_file

img_disp = []
df_list = []

def getProductData(img_file):
    global category
    for x in img_file:
        if x[0] == "0":
            img_dir = 'dataset/cleaned_Bathroom Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Bathroom/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "1":
            img_dir = 'dataset/cleaned_Bedroom Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Bedroom/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "2":
            img_dir = 'dataset/cleaned_Diningroom Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Diningroom/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "3":
            img_dir = 'dataset/cleaned_Gameroom Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Gameroom/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "4":
            img_dir = 'dataset/cleaned_HomeOffice Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/HomeOffice/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "5":
            img_dir = 'dataset/cleaned_Kids Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Kids/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "6":
            img_dir = 'dataset/cleaned_Kitchen Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Kitchen/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "7":
            img_dir = 'dataset/cleaned_Livingroom Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/Livingroom/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)


        elif x[0] == "8":
            img_dir = 'dataset/cleaned_Replacementparts Furniture.xlsx'
            new_df = readDataFrame(img_dir, x)
            img_dir = 'images3/ReplacementParts/'
            img_path = '{}{}'.format(img_dir, x)
            category = img_dir.split('/')[1].strip()
            img = cv2.imread(img_path)
            img_disp.append(img)
            df_list.append(new_df)

    return category,img_disp,df_list


stop_words = set(stopwords.words('english'))
titlepreprocess_list = []
titlepreprocess_list1 = []
def nlp_preprocessing(total_text, index,column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split(): #split out the total text by space
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum())) #remove special characters by checking single word inside
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        df_titlepreprocess = pd.DataFrame()
        df_titlepreprocess[column] = [string] #append titlte into dataframe
        df_titlepreprocess.index = [index] # change index
        titlepreprocess_list.append(df_titlepreprocess) #list of dataframes

def nlp_preprocessing1(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():  # split out the total text by space
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = (
                "".join(e for e in words if e.isalnum()))  # remove special characters by checking single word inside
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        df_tempnlp = pd.DataFrame()
        df_tempnlp[column] = [string]  # append titlte into dataframe
        df_tempnlp.index = [index]  # change index
        titlepreprocess_list1.append(df_tempnlp) # list of dataframes

listofvalues = []
listoflabels = []
listofkeys = []


def plot_heatmap_image( vec1, vec2, text, model):
    # vec1 : input furniture's vector, it is of a dict type {word:count}
    # vec2 : recommended furniture's vector, it is of a dict type {word:count}
    # text: title of recomonded furniture
    # model, it can be any of the models,
    # 1. bag_of_words
    # 2. tfidf
    # we find the common words in both titles, because these only words contribute to the distance between two title vec's
    intersection = set(vec1.keys()) & set(vec2.keys())
    # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
    for i in vec2:
        if i not in intersection:
            vec2[i] = 0
    # for labeling heatmap, keys contains list of all words in title2
    keys = list(vec2.keys())
    #  if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else values(i)=0
    values = [vec2[x] for x in vec2.keys()]
    # labels: len(labels) == len(keys), the values of labels depends on the model we are using
    # if model == 'bag of words': labels(i) = values(i)
    # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[20, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
        # for x in vec2.keys():
        #     # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
        #     # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
        #     if x in tfidf_title_vectorizer.vocabulary_:
        #         labels.append(tfidf_title_features[ tfidf_title_vectorizer.vocabulary_[x]])
        #     else:
        #         labels.append(0)

    listofvalues.append(values)
    listoflabels.append(labels)
    listofkeys.append(keys)


# this function gets a list of wrods along with the frequency of each word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()'
    # this will also gives same result
    return Counter(words)  # Counter counts the occurence of each word in list,
    # it returns dict type object {word1:count}


def get_result(content_a, content_b, model):
    text1 = content_a   #input from user
    text2 = content_b   #all other furniture (each)
    vector1 = text_to_vector(text1) # vector1 = dict{word11:#count, word12:#count, etc.}
    vector2 = text_to_vector(text2) # vector1 = dict{word21:#count, word22:#count, etc.}
    plot_heatmap_image( vector1, vector2,text2, model)


def tfidf_model(num_results):
    # pairwise_dist will store the distance from given input furniture to all furniture in the database
    pairwise_dist = pairwise_distances(tfidf_title_features[:20], tfidf_title_features[20],metric='cosine')
    pairwise_trans = pairwise_dist.T
    # np.argsort will return indices of the smallest distances
    # here is follow the numbering 0 - end
    len_product = len(pairwise_dist.flatten())
    #indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    # pdists will store the smallest distances
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]

    #     bag_of_words_euclidean.append(pdists[i])
    #
    # print('Average euclidean distance is ', sum(bag_of_words_euclidean) / num_results)
    return pairwise_trans, len_product


def top5(img_disp,df_list,listofvalues=None,listoflabels=None,listofkeys=None):
    recomm = []
    for x in range(5):
        recomm.append("Recommendation " + str(x + 1))

    if bool(img_disp):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(recomm)
        with tab1:
            st.header(df_list[0].loc[:, "Title"].item())
            st.image(img_disp[0], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
                ax = sns.heatmap(np.array([listofvalues[0]]), annot=np.array([listoflabels[0]]))
                ax.set_xticklabels(listofkeys[0])  # set that axis labels as the words of title
                text = ' '.join(map(str,listofkeys[0]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)

            st.write("URL: " + df_list[0].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[0].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[0].loc[:, "Price"].item())
        with tab2:
            st.header(df_list[1].loc[:, "Title"].item())
            st.image(img_disp[1], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[1]]), annot=np.array([listoflabels[1]]))
                ax.set_xticklabels(listofkeys[1])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[1]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[1].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[1].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[1].loc[:, "Price"].item())
        with tab3:
            st.header(df_list[2].loc[:, "Title"].item())
            st.image(img_disp[2], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[2]]), annot=np.array([listoflabels[2]]))
                ax.set_xticklabels(listofkeys[2])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[2]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[2].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[2].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[2].loc[:, "Price"].item())
        with tab4:
            st.header(df_list[3].loc[:, "Title"].item())
            st.image(img_disp[3], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[3]]), annot=np.array([listoflabels[3]]))
                ax.set_xticklabels(listofkeys[3])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[3]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[3].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[3].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[3].loc[:, "Price"].item())
        with tab5:
            st.header(df_list[4].loc[:, "Title"].item())
            st.image(img_disp[4], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[4]]), annot=np.array([listoflabels[4]]))
                ax.set_xticklabels(listofkeys[4])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[4]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[4].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[4].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[4].loc[:, "Price"].item())

def top8(img_disp,df_list,listofvalues=None,listoflabels=None,listofkeys=None):
    recomm = []
    for x in range(8):
        recomm.append("Recommendation " + str(x + 1))

    if bool(img_disp):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(recomm)
        with tab1:
            st.header(df_list[0].loc[:, "Title"].item())
            st.image(img_disp[0], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
                ax = sns.heatmap(np.array([listofvalues[0]]), annot=np.array([listoflabels[0]]))
                ax.set_xticklabels(listofkeys[0])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[0]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)

            st.write("URL: " + df_list[0].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[0].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[0].loc[:, "Price"].item())
        with tab2:
            st.header(df_list[1].loc[:, "Title"].item())
            st.image(img_disp[1], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[1]]), annot=np.array([listoflabels[1]]))
                ax.set_xticklabels(listofkeys[1])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[1]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[1].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[1].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[1].loc[:, "Price"].item())
        with tab3:
            st.header(df_list[2].loc[:, "Title"].item())
            st.image(img_disp[2], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[2]]), annot=np.array([listoflabels[2]]))
                ax.set_xticklabels(listofkeys[2])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[2]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[2].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[2].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[2].loc[:, "Price"].item())
        with tab4:
            st.header(df_list[3].loc[:, "Title"].item())
            st.image(img_disp[3], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[3]]), annot=np.array([listoflabels[3]]))
                ax.set_xticklabels(listofkeys[3])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[3]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[3].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[3].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[3].loc[:, "Price"].item())
        with tab5:
            st.header(df_list[4].loc[:, "Title"].item())
            st.image(img_disp[4], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[4]]), annot=np.array([listoflabels[4]]))
                ax.set_xticklabels(listofkeys[4])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[4]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[4].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[4].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[4].loc[:, "Price"].item())
        with tab6:
            st.header(df_list[5].loc[:, "Title"].item())
            st.image(img_disp[5], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[5]]), annot=np.array([listoflabels[5]]))
                ax.set_xticklabels(listofkeys[5])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[5]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[5].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[5].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[5].loc[:, "Price"].item())
        with tab7:
            st.header(df_list[6].loc[:, "Title"].item())
            st.image(img_disp[6], width=250)
            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[6]]), annot=np.array([listoflabels[6]]))
                ax.set_xticklabels(listofkeys[6])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[6]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[6].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[6].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[6].loc[:, "Price"].item())
        with tab8:
            st.header(df_list[7].loc[:, "Title"].item())
            st.image(img_disp[7], width=250)
            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[7]]), annot=np.array([listoflabels[7]]))
                ax.set_xticklabels(listofkeys[7])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[7]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[7].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[7].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[7].loc[:, "Price"].item())

def top10(img_disp,df_list,listofvalues=None,listoflabels=None,listofkeys=None):
    recomm = []
    for x in range(10):
        recomm.append("Recommendation " + str(x + 1))

    if bool(img_disp):
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(recomm)
        with tab1:
            st.header(df_list[0].loc[:, "Title"].item())
            st.image(img_disp[0], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
                ax = sns.heatmap(np.array([listofvalues[0]]), annot=np.array([listoflabels[0]]))
                ax.set_xticklabels(listofkeys[0])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[0]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)

            st.write("URL: " + df_list[0].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[0].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[0].loc[:, "Price"].item())
        with tab2:
            st.header(df_list[1].loc[:, "Title"].item())
            st.image(img_disp[1], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[1]]), annot=np.array([listoflabels[1]]))
                ax.set_xticklabels(listofkeys[1])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[1]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[1].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[1].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[1].loc[:, "Price"].item())
        with tab3:
            st.header(df_list[2].loc[:, "Title"].item())
            st.image(img_disp[2], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[2]]), annot=np.array([listoflabels[2]]))
                ax.set_xticklabels(listofkeys[2])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[2]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[2].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[2].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[2].loc[:, "Price"].item())
        with tab4:
            st.header(df_list[3].loc[:, "Title"].item())
            st.image(img_disp[3], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[3]]), annot=np.array([listoflabels[3]]))
                ax.set_xticklabels(listofkeys[3])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[3]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[3].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[3].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[3].loc[:, "Price"].item())
        with tab5:
            st.header(df_list[4].loc[:, "Title"].item())
            st.image(img_disp[4], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[4]]), annot=np.array([listoflabels[4]]))
                ax.set_xticklabels(listofkeys[4])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[4]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[4].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[4].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[4].loc[:, "Price"].item())
        with tab6:
            st.header(df_list[5].loc[:, "Title"].item())
            st.image(img_disp[5], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[5]]), annot=np.array([listoflabels[5]]))
                ax.set_xticklabels(listofkeys[5])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[5]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[5].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[5].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[5].loc[:, "Price"].item())
        with tab7:
            st.header(df_list[6].loc[:, "Title"].item())
            st.image(img_disp[6], width=250)
            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[6]]), annot=np.array([listoflabels[6]]))
                ax.set_xticklabels(listofkeys[6])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[6]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[6].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[6].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[6].loc[:, "Price"].item())
        with tab8:
            st.header(df_list[7].loc[:, "Title"].item())
            st.image(img_disp[7], width=250)
            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[7]]), annot=np.array([listoflabels[7]]))
                ax.set_xticklabels(listofkeys[7])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[7]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[7].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[7].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[7].loc[:, "Price"].item())
        with tab9:
            st.header(df_list[8].loc[:, "Title"].item())
            st.image(img_disp[8], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[8]]), annot=np.array([listoflabels[8]]))
                ax.set_xticklabels(listofkeys[8])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[8]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[8].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[8].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[8].loc[:, "Price"].item())
        with tab10:
            st.header(df_list[9].loc[:, "Title"].item())
            st.image(img_disp[9], width=250)

            if listofvalues is not None:
                fig, ax = plt.subplots(figsize=(30, 3))
                ax = sns.heatmap(np.array([listofvalues[9]]), annot=np.array([listoflabels[9]]))
                ax.set_xticklabels(listofkeys[9])  # set that axis labels as the words of title
                text = ' '.join(map(str, listofkeys[9]))
                ax.set_title(text)  # apparel title
                st.pyplot(fig)
            st.write("URL: " + df_list[8].loc[:, "Url"].item())
            st.write("Average rating: " + df_list[8].loc[:, "Avg_ratings"].item())
            st.write("Price: " + df_list[8].loc[:, "Price"].item())





upload_file = st.file_uploader("First step: Select a Furniture Image to upload",type=['png','jpeg','jpg'])
img_file = []

df_textinput = pd.DataFrame()


if upload_file is not None:
    #display_image = Image.open(upload_file)
    #st.image(display_image)

    #save_upload_file(upload_file) #local database
    #features = feature_extraction(os.path.join("uploads", upload_file.name), model) #local database


    img = Image.open(upload_file) #clouds
    features = feature_extraction(img,model) #clouds

    maxindex = features.argmax()
    maxvalues = np.max(features)
    img_file = append_imgfilename(maxindex)
    category, img_disp, df_list = getProductData(img_file)



    font_size = 20
    n1 = "\n"
    category_accuracy = str(maxvalues * 100) + "%"

    html_str = f"""
    <style>
    p.a {{
      font: bold {font_size}px Courier;
    }}
    </style>
    <p class="a">{"The predicted category: " + category }</p>
    """
    
    accuracy_text = f"""
        <style>
        p.a {{
          font: bold {font_size}px Courier;
        }}
        </style>
        <p class="a">{"Accuracy: " + category_accuracy}</p>
        """

    st.markdown(html_str, unsafe_allow_html=True)
    st.markdown(accuracy_text, unsafe_allow_html=True)

    st.write("\n")
    option = st.selectbox('Do you want to enter the product name for more accurate results?', ('None', 'Yes'))

    if option == 'Yes':
        prod_title = st.text_input('Please enter the name on below')
        df_textinput["Title"] = [prod_title]

        df_list.append(df_textinput)

        for x in range(len(df_list)):
            title = df_list[x].loc[:, "Title"].item()
            index = df_list[x].index[0]
            nlp_preprocessing(title, index, "Title")

        df_concat = pd.concat(titlepreprocess_list)

        tfidf_title_vectorizer = TfidfVectorizer(min_df=0)
        tfidf_title_features = tfidf_title_vectorizer.fit_transform(df_concat['Title'])
        tfidf_title_features.get_shape()  # get number of rows and columns in feature matrix.
        # title_features.shape = #data_points * #words_in_corpus
        # CountVectorizer().fit_transform(corpus) returns
        # the a sparase matrix of dimensions #data_points * #words_in_corpus
        tf_idf_euclidean=[]

        title_distance, len_product = tfidf_model(20)
        img_file = append_imgfilename(maxindex,title_distance)
        img_disp = []
        df_list = []
        category, img_disp2, df_list2 = getProductData(img_file)
        df_temp = pd.concat(df_list2, axis=0)
        df_temp = df_temp[['Title']]
        for x in range(len(df_temp)):
            title = df_temp["Title"].iloc[x]
            temp = df_temp.index
            index = temp[x]
            nlp_preprocessing1(title, index, "Title")

        titlepreprocess_list1 = pd.concat(titlepreprocess_list1, axis=0)
        for i in range(0, (len_product - 1)):
            # we will pass the input, all other furniture, model
            get_result(df_concat.iloc[-1, 0], titlepreprocess_list1.iloc[i, 0], 'tfidf')

        choices = st.radio("Second step: Choose top k recommendation",("TOP 5","TOP 8","TOP 10"))




        if st.button("Click here to get recommendations!"):
            st.success('Hope you like the below recommendations :)')
            if choices == 'TOP 5':
                top5(img_disp2,df_list2,listofvalues,listoflabels,listofkeys)

            elif choices == 'TOP 8':
                top8(img_disp2,df_list2,listofvalues,listoflabels,listofkeys)

            elif choices == 'TOP 10':
                top10(img_disp2,df_list2,listofvalues,listoflabels,listofkeys)

    elif option == "None":

        choices = st.radio("Second step: Choose top k recommendation", ("TOP 5", "TOP 8", "TOP 10"))

        if st.button("Click here to get recommendations!"):
            st.success('Hope you like the below recommendations :)')
            if choices == 'TOP 5':
                top5(img_disp, df_list)

            elif choices == 'TOP 8':
                top8(img_disp, df_list)

            elif choices == 'TOP 10':
                top10(img_disp, df_list)










