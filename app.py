import tensorflow

import tensorflow as tf
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import pathlib
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras.optimizers import adam_v2

# data_dir = pathlib.Path(r'images2') # represent a path to a file or directory
# train_dir = pathlib.Path(r'furniture-2022/training')
# val_dir = pathlib.Path(r'furniture-2022/validation')
# roses = list(data_dir.glob('Bathroom/*'))
#
# img_height,img_width=224,224
# batch_size=32
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   train_dir,
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   val_dir,
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# model = Sequential()
#
# pretrained_model= ResNet50(include_top=False,
#                    input_shape=(224,224,3),
#                    pooling='avg',classes=9,
#                    weights='imagenet')
# for layer in pretrained_model.layers:
#         layer.trainable=False
#
# model.add(pretrained_model)
# model.add(Flatten())
#
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
#
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
#
# model.add(Dense(9, activation='softmax'))
#
# model.compile(optimizer=adam_v2.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# epochs=2
# history = model.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

model = load_model("imgaug_rmspropV2_Ortho_2.h5")

def extract_features(img_path,model):
    if img_path[0] == "0":
        img_dir = 'images3/Bathroom/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "1":
        img_dir = 'images3/Bedroom/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "2":
        img_dir = 'images3/Diningroom/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "3":
        img_dir = 'images3/Gameroom/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "4":
        img_dir = 'images3/HomeOffice/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "5":
        img_dir = 'images3/Kids/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "6":
        img_dir = 'images3/Kitchen/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "7":
        img_dir = 'images3/Livingroom/'
        full_path = '{}{}'.format(img_dir, img_path)
    elif img_path[0] == "8":
        img_dir = 'images3/ReplacementParts/'
        full_path = '{}{}'.format(img_dir, img_path)

    img = image.load_img(full_path,target_size=(224,224)) # load image in PIL format (widthXheightXchannels)
    img_array = image.img_to_array(img) # convert into numpy format (heightXwidthXchannels)
    expanded_img_array = np.expand_dims(img_array, axis=0) # convert into batch format (batchsizeXheightXweightXchannel)
    # have extra dimensions
    preprocessed_img = preprocess_input(expanded_img_array)
    preprocessed_img = preprocessed_img/255
    result = model.predict(preprocessed_img).flatten() # get classification results
    normalized_result = result / norm(result)

    return normalized_result
filenames = []
filenames_Bathroom= []
filenames_Bedroom= []
filenames_Diningroom= []
filenames_Gameroom= []
filenames_HomeOffice= []
filenames_Kids= []
filenames_Kitchen= []
filenames_Living= []
filenames_Replace= []

for file in os.listdir('images3/'): #get the list of all files and directories in the specified directory
    path = 'images3/'
    file = os.path.join(path, file)
    for file in os.listdir(file):
        filenames.append(file)


for file in os.listdir('images3/'): #get the list of all files and directories in the specified directory
    path = 'images3/'
    file = os.path.join(path, file)
    if file =="images3/Bathroom":
        for file in os.listdir(file):
            filenames_Bathroom.append(file)
    elif file =="images3/Bedroom":
        for file in os.listdir(file):
            filenames_Bedroom.append(file)
    elif file =="images3/Diningroom":
        for file in os.listdir(file):
            filenames_Diningroom.append(file)
    elif file =="images3/Gameroom":
        for file in os.listdir(file):
            filenames_Gameroom.append(file)
    elif file =="images3/HomeOffice":
        for file in os.listdir(file):
            filenames_HomeOffice.append(file)
    elif file =="images3/Kids":
        for file in os.listdir(file):
            filenames_Kids.append(file)
    elif file =="images3/Kitchen":
        for file in os.listdir(file):
            filenames_Kitchen.append(file)
    elif file =="images3/Livingroom":
        for file in os.listdir(file):
            filenames_Living.append(file)
    elif file =="images3/ReplacementParts":
        for file in os.listdir(file):
            filenames_Replace.append(file)


feature_list_Bathroom = []
feature_list_Bedroom = []
feature_list_Dining = []
feature_list_Game = []
feature_list_HomeOffice = []
feature_list_Kids = []
feature_list_Kitchen = []
feature_list_Living = []
feature_list_Replacement = []

for file in tqdm(filenames):
    feature = extract_features(file,model)
    if file[0] == "0":
        feature_list_Bathroom.append(feature)
    elif file[0] == "1":
        feature_list_Bedroom.append(feature)
    elif file[0] == "2":
        feature_list_Dining.append(feature)
    elif file[0] == "3":
        feature_list_Game.append(feature)
    elif file[0] == "4":
        feature_list_HomeOffice.append(feature)
    elif file[0] == "5":
        feature_list_Kids.append(feature)
    elif file[0] == "6":
        feature_list_Kitchen.append(feature)
    elif file[0] == "7":
        feature_list_Living.append(feature)
    elif file[0] == "8":
        feature_list_Replacement.append(feature)
    # feature_list.append(extract_features(file,model))

pickle.dump(feature_list_Bathroom,open('embeddings_Bathroom.pkl','wb'))
pickle.dump(feature_list_Bedroom,open('embeddings_Bedroom.pkl','wb'))
pickle.dump(feature_list_Dining,open('embeddings_Dining.pkl','wb'))
pickle.dump(feature_list_Game,open('embeddings_Game.pkl','wb'))
pickle.dump(feature_list_HomeOffice,open('embeddings_HomeOffice.pkl','wb'))
pickle.dump(feature_list_Kids,open('embeddings_Kids.pkl','wb'))
pickle.dump(feature_list_Kitchen,open('embeddings_Kitchen.pkl','wb'))
pickle.dump(feature_list_Living,open('embeddings_Living.pkl','wb'))
pickle.dump(feature_list_Replacement,open('embeddings_Replacement.pkl','wb'))

pickle.dump(filenames,open('filenames.pkl','wb'))
pickle.dump(filenames_Bathroom,open('filenames_Bath.pkl','wb'))
pickle.dump(filenames_Bedroom,open('filenames_Bed.pkl','wb'))
pickle.dump(filenames_Diningroom,open('filenames_Dining.pkl','wb'))
pickle.dump(filenames_Gameroom,open('filenames_Game.pkl','wb'))
pickle.dump(filenames_HomeOffice,open('filenames_Home.pkl','wb'))
pickle.dump(filenames_Kids,open('filenames_Kids.pkl','wb'))
pickle.dump(filenames_Kitchen,open('filenames_Kitchen.pkl','wb'))
pickle.dump(filenames_Living,open('filenames_Living.pkl','wb'))
pickle.dump(filenames_Replace,open('filenames_Replace.pkl','wb'))
