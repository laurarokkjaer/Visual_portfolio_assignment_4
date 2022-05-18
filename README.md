
# Visual Analytics - Spring 2022
# Portfolio Assignment 4 - MY SELF-ASSIGNED PROJECT

This repository contains the code and descriptions from the last self-assigned project of the Spring 2022 module Visual Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Visual analytics portfolio (zip-file) consist of 4 projects, 3 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```input``` | Contains the input data (will be empty) |
| ```output``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for the self-assigned assignment 4 |
| ```utils``` | Contains utility functions written by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and which have been used in the assignments |

Also containing a ```MITLICENSE``` for guidelines of how to reproduce and use the data in this repository, as well as a ```.txt``` reqirements-file, where the required installments will be listed.

## Assignment description
This is my self-assigned project for which i have chosen to solve the following task:
How to train a model to predict/detect emotions in facial expression images with a defined neural network architecture using the deep learning method ```tf.keras```. The task is then to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
To solve this assignment i will do the following:
- Download data, train-test split, binarize labels
- Define neural network model
- Compile model loss function, optimizer, and preferred metrics
- Train model and save history
- Visualise loss/accuracy and evaluate network with a classification report 


### The goal of the assignment 
The goal of this assignment was to demonstrate my knowledge within deep learning models used this semester, along with the building of a model which can predict facial expressions in images.

### Data source
The data used in this assignment is from [kaggle](https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb#scrollTo=59mL7DzN139i) (FER2013 dataset). 

Download data from command line: 
wget -O ../in/fer2013.csv https://www.dropbox.com/s/zi48lkarsg4kbry/fer2013.csv\?dl\=1

Official data description from above link: 
The csv file contains two main columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.
This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.


## Methods
Frist of all I reused a helping function from [kaggle](https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb#scrollTo=59mL7DzN139i), which helps loading the data as well as splits it into train_test (only a few adjustments in order to get it to run with my choise of modelling methods). 
The methods used is ```tensorflow``` operations like ```ImageDataGenerator```, ```load_img``` and ```img_to_array``` in terms of preprocessing, as well as initializing the model ```Sequential()``` and it's layers. Futhermore, using ```scikit-learn```for ```LabelBinarizer``` and ```classification_report```, and at last ```matplotlib``` for visualisations.


## Usage (reproducing results)
These are the steps you will need to follow in order to get the script running and working:
- load the given data into ```input```
- make sure to install and import all necessities from ```requirements.txt``` 
- change your current working directory to the folder before src in order to get access to the input, output and utils folder as well 
- the following should be written in the command line:

      - cd src (changing the directory to the src folder in order to run the script)
      
      - python self_assigned.py (calling the function within the script)
      
- when processed results there will be a messagge saying that the script has succeeded and the outputs can be seen in the output folder 


## Discussion of results
The result of the script is a classification accuracy of 34%, but it would have been interesting to see if the accuracy could get any higher if another deep learning model was used. So for further development, one could try with the VGG16 model perhabs. Only this would require some CNN skills to fit a new shape of the data, since the dataset is greyscale images and the VGG16 only work with colour images where you have all three colour channels. 
