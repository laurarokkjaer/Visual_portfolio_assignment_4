import os 
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# data processing
import pandas as pd 
# layers
from tensorflow import keras
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt


def emotion_detection():
    def load_dataset(net=True):
        """Utility function to load the FER2013 dataset.

        It returns the formated tuples (X_train, y_train) , (X_test, y_test).

        Parameters
        ==========
        net : boolean
            This parameter is used to reshape the data from images in 
            (cols, rows, channels) format. In case that it is False, a standard
            format (cols, rows) is used.
        """

        # Load and filter in Training/not Training data:
        df = pd.read_csv('Input/fer2013.csv')
        training = df.loc[df['Usage'] == 'Training']
        print(training.shape)
        testing = df.loc[df['Usage'] != 'Training']
        print(testing.shape)

        # X_train values:
        X_train = training[['pixels']].values
        X_train = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_train]
        if net:
            X_train = [e.reshape((48, 48, 1)).astype('float32').flatten() for e in X_train]
        else:
            X_train = [e.reshape((48, 48)) for e in X_train]
        X_train = np.array(X_train)

        # X_test values:
        X_test = testing[['pixels']].values
        X_test = [np.fromstring(e[0], dtype=int, sep=' ') for e in X_test]
        if net:
            X_test = [e.reshape((48, 48, 1)).astype('float32').flatten() for e in X_test]
        else:
            X_test = [e.reshape((48, 48)) for e in X_test]
        X_test = np.array(X_test)

        # y_train values:
        y_train = training[['emotion']].values
        y_train = keras.utils.to_categorical(y_train)

        # y_test values
        y_test = testing[['emotion']].values
        y_test = keras.utils.to_categorical(y_test)

        return (X_train, y_train) , (X_test, y_test)
    
    (X_train, y_train) , (X_test, y_test) = load_dataset()
    
    labels = ["angry",
          "disgusted",
         "fearful",
         "happy",
         "netraul",
         "sad",
         "surprised",
         ]
    
    # Binarize labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    #Normalize data
    X_train_norm = X_train/255
    X_test_norm = X_test/255
    
    # Define model architecture
    model = Sequential()
    model.add(Dense(256, 
                    activation = "relu", 
                    input_shape=(2304,)))
    model.add(Dense(128, 
                    activation = "relu"))
    model.add(Dense(7, 
                    activation = "softmax"))
    
    # Training
    # Define the gradient descent 
    sgd = SGD(0.001)
    # Compile model (defining loss function, the optimizer, metrics)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = sgd,
                  metrics = ["accuracy"])
    
    history = model.fit(X_train_norm, y_train,
                   validation_data = (X_test_norm, y_test), 
                   epochs = 10,
                   batch_size = 32)
    
    # Defining the colorpalat as my visualization style 
    plt.style.use("seaborn-colorblind")

    # Create canvas for loss
    plt.figure(figsize=(12,6))
    # A subplot saying that on the y-axis i want 1 row and on the x-axis 2 columns and on the first image 
    plt.subplot(1,2,1)
    # Plotting my loss from 0-10 (epochs) giving it a label 
    # Both the training loss and the validation loss (val_loss)
    plt.plot(np.arange(0, 10), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 10), history.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    # Same for accuracy
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, 10), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 10), history.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    # Evaluate network 
    predictions = model.predict(X_test, batch_size = 32)
    report = classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels)
    
    # Using .write to save the report in a .txt file in my outputfolder
    with open('output/emotions_report.txt', 'w') as my_txt_file:
        my_txt_file.write(report)
    
    print("The following result is the classification report for my emotion detectio model. The results can be seen in the output-folder as well")
    print(report)
    
emotion_detection()
    
    

    
