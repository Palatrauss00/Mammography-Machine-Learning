import os
import glob
import cv2
import numpy
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from keras.preprocessing.image import image
from keras.models import Sequential, Model
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
import keras.layers
import keras.optimizers
import keras
from keras.optimizer_v2.rmsprop import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
import time

def load_validation_set():
    healthy_case_dir = val_Dir/'Healthy'
    non_healthy_case_dir = val_Dir/'NotHealthy'
    healthy_case = healthy_case_dir.glob('*.jpg')
    non_healthy_case = non_healthy_case_dir.glob('*.jpg')
    train_data =[]
    train_label = []
    for img in healthy_case:
        train_data.append(str(img))
        train_label.append('HEALTHY')
    for img in non_healthy_case:
        train_data.append(str(img))
        train_label.append('NON HEALTHY')
    df = pd.DataFrame(train_data)
    df.columns = ['Images']
    df['labels'] = train_label
    df = df.sample(frac =1).reset_index(drop=True)
    return df

def load_training_set():
    healthy_case_dir = train_Dir/'Healthy'
    non_healthy_case_dir = train_Dir/'NotHealthy'
    healthy_case = healthy_case_dir.glob('*.jpg')
    non_healthy_case = non_healthy_case_dir.glob('*.jpg')
    train_data =[]
    train_label = []
    for img in healthy_case:
        train_data.append(str(img))
        train_label.append('HEALTHY')
    for img in non_healthy_case:
        train_data.append(str(img))
        train_label.append('NON HEALTHY')
    df = pd.DataFrame(train_data)
    df.columns = ['Images']
    df['labels'] = train_label
    df = df.sample(frac =1).reset_index(drop=True)
    return df

def plot_graph(log_data):
    plt.plot(log_data['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(log_data['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
	#Set toTrain to True to train a new model
    toTrain = False



    if toTrain == True:
        start_time = time.time()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        datasetDirectory = Path("MAIN_PATH")
        train_Dir = datasetDirectory/'training'
        val_Dir  = datasetDirectory/'validation'
        test_Dir = datasetDirectory/'testing'
        train_dataframe= load_training_set()
        val_dataframe = load_validation_set()
        plt.bar(train_dataframe['labels'].value_counts().index, train_dataframe['labels'].value_counts().values)
        plt.show()
        #Operation on dataset
        train_data_gen = ImageDataGenerator(rescale=1/255)
        val_data_gen = ImageDataGenerator(rescale=1/255)
        training_dataset = train_data_gen.flow_from_dataframe(
            dataframe=train_dataframe,
            directory=train_Dir,
            x_col='Images',
            y_col='labels',
            subset='training',
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode='binary',
            target_size=(300,300),
            validate_filenames=False
        )
        val_dataset = val_data_gen.flow_from_dataframe(
            dataframe=val_dataframe,
            directory=val_Dir,
            x_col='Images',
            y_col='labels',
            subset='validation',
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode='binary',
            target_size=(300, 300),
            validate_filenames=False
        )
        print(training_dataset.class_indices)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
        #
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
        #
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),

            tf.keras.layers.Flatten(),
        #
            tf.keras.layers.Dense(512, activation='relu'),
        #
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])

        csv_logger = CSVLogger('training.log', separator=',', append=False)
        model_fit = model.fit(training_dataset,
                          epochs=50,  
                          validation_data=val_dataset,  callbacks=[csv_logger])

        model.save("PATH_FOR_SAVING_MODEL")
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        model = tf.keras.models.load_model(
            "PATH_SAVED_MODEL")
        img1 = image.load_img(
            "IMAGE_PATH",
            target_size=(300, 300))
        imgArr = image.img_to_array(img1)
        imgArr = np.expand_dims(imgArr, axis=0)
        # images = np.stack([imgArr])
        val = model.predict(imgArr)
        if val == 0:
            print("HEALTHY")

        else:
            print("NON-HEALTHY")


        log_data = pd.read_csv('training.log', sep=',', engine='python')
        plot_graph(log_data)
