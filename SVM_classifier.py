# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import  numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #Change this to True to load new Data
    newDataset = False
    if(newDataset == True):
        dir = "DIRECTORY_PATH"
        categories = ['HEALTHY', 'NOTHEALTHY']
        data = []

        for category in categories:
            path=  os.path.join(dir,category)
            label = categories.index(category)
            for img in os.listdir(path):
                imgpath = os.path.join(path,img)
                myimg = cv2.imread(imgpath,0)
                try:
                    myimg = cv2.resize(myimg,(100,100))
                    image = np.array(myimg).flatten()
                    data.append([image,label])
                except Exception as e:
                    pass
        #Saves Dataset in Current Folder
        pickle_in = open('data1.pickle', 'wb')
        pickle.dump(data,pickle_in)
        pickle_in.close()


    else:

        pickle_in = open('data1.pickle', 'rb')
        data = pickle.load(pickle_in)
        pickle_in.close()

    random.shuffle(data)
    features =[]
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(features,labels, test_size=0.5)

    newModel = True
    if(newModel == True):
    # Change this to True to load new Data
        model = SVC(C=1,kernel='poly',gamma='auto', probability=True, random_state=42)
        model.fit(xtest,ytrain)

        pickle_m = open('model.sav','wb')
        pickle.dump(model, pickle_m)
        pickle_m.close()

    else:
        pickle_m = open('model.sav', 'rb')
        model = pickle.load(pickle_m)
        pickle_m.close()

    categories = ['HEALTHY', 'NOTHEALTHY']
    predictions = model.predict(xtrain)
    accuracy = model.score(xtrain,ytrain)

    print('Accuracy: ', accuracy)
    print('Predizione: ', categories[predictions[0]])

    myimg = xtest[0].reshape(100,100)
    plt.imshow(myimg,cmap='gray')
    plt.show()
    plt.close()

    probabilities = model.predict_proba(xtest)

    # select the probabilities for label 1.0
    y_proba = probabilities[:, 1]

    # calculate false positive rate and true positive rate at different thresholds
    false_positive_rate, true_positive_rate, thresholds = roc_curve(ytest, y_proba, pos_label=1)

    # calculate AUC
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    # plot the false positive rate on the x axis and the true positive rate on the y axis
    roc_plot = plt.plot(false_positive_rate,
                        true_positive_rate,
                        label='AUC = {:0.2f}'.format(roc_auc))

    plt.legend(loc=0)
    plt.plot([0, 1], [0, 1], ls='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate');
    plt.show()
    plt.close()
    #Change to True To Test On your Image
    testMyImage = False

    if(testMyImage == True):
        myimgPath = 'IMAGE_PATH'
        myimg_1 = cv2.imread(myimgPath, 0)
        myimg_1 =cv2.resize(myimg_1, (100, 100))
        image1 = np.array(myimg_1).flatten()
        testImg = []
        testImg.append(image1)
        predictions_New = model.predict(testImg)
        print('Predizione Immagine: ', categories[predictions_New[0]])
        myimg_1 = testImg[0].reshape(100,100)
        plt.imshow(myimg_1,cmap='gray')
        plt.show()

