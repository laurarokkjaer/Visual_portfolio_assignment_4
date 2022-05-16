# Visual Analytics - Spring 2022
# Portfolio Assignment 4 - MY SELF-ASSIGNED PROJECT

This repository contains the code and descriptions from the last self-assigned project of the Spring 2022 module Visual Analytics as part of the bachelor's tilvalg in Cultural Data Science at Aarhus University - whereas the overall Visual analytics portfolio (zip-file) consist of 4 projects, 3 class assignments + 1 self-assigned.

## Repo structure
### This repository has the following directory structure:

| **Folder** | **Description** |
| ----------- | ----------- |
| ```in``` | Contains the input data (will be empty) |
| ```out``` | Contains the results (outputs like plots or reports)  |
| ```src``` | Contains code for assignment 1 |
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
The data used in this assignment is from [kaggle](https://colab.research.google.com/github/RodolfoFerro/PyConCo20/blob/full-code/notebooks/Deep%20Learning%20Model.ipynb#scrollTo=59mL7DzN139i). 
Official data description from above link: 
The csv file contains two main columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. The "pixels" column contains a string surrounded in quotes for each image. The contents of this string a space-separated pixel values in row major order. test.csv contains only the "pixels" column and your task is to predict the emotion column.
This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project. They have graciously provided the workshop organizers with a preliminary version of their dataset to use for this contest.

Download data from command line: 
wget -O ../in/fer2013.csv https://www.dropbox.com/s/zi48lkarsg4kbry/fer2013.csv\?dl\=1


## Methods
To solve this assignment i have worked with ```opencv``` in order to both calculate the histograms as well as for the general image processing (using the ```calcHist```, ```imread```, ```normalize``` and ```compareHist```). Futhermore i used the ```jimshow``` and ```jimshow_channel``` from the ```utils```-folder, along with the ```matplotlib``` for plotting and visualisation.

## Usage (reproducing results)
For this .py script the following should be written in the command line:
- change directory to the folder /src 
- write command: python image_search.py
- when processing results there will be a messagge saying that the script has succeeded and the outputs can be seen in the output folder 

The target image, as well as the most similar images can be seen in the output folder both as csv (with file informations) and as a visualisation where the images are plottet next to each other


## Discussion of results
something about 
- a user defined input (what that could do for the assignment and the reproducability 
- the transision from a notebook to a .py script 

