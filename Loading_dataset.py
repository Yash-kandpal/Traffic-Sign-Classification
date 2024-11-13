import zipfile
import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

zip_filepath = 'C:/Users/yashk/Downloads/data3.pickle (1).zip'
extractdir = 'C:/Users/yashk/PycharmProjects/Traffic-Sign-Recognition'

with zipfile.ZipFile(zip_filepath,'r') as zip_ref:
    zip_ref.extractall(extractdir)

print("Datafile unzipped....Successfully")

pickle_filepath = os.path.join(extractdir,'data3.pickle')

with open(pickle_filepath,'rb') as file:
    data = pickle.load(file)

print(type(data))
print(data.keys(),"\n")

train_images = data.get('x_train')
test_images = data.get('x_test')
valdn_images = data.get('x_validation')
valdn_labels = data.get('y_validation')
test_labels = data.get('y_test')
train_labels = data.get('y_train')

# Using NumPy to transpose if images are NumPy arrays
train_images = np.transpose(train_images, (0, 2, 3, 1))
valdn_images = np.transpose(valdn_images, (0, 2, 3, 1))
test_images = np.transpose(test_images, (0, 2, 3, 1))


print("Train images : ",train_images.shape,"\t Validation Images : ",valdn_images.shape,"\t Test Images : ",test_images.shape,"\n")
print("Train labels : ",train_labels.shape,"\t validation labels : ",valdn_labels.shape,"\t test labels : ",test_labels.shape)
num_classes = len(set(train_labels))
print("Number of classes :", num_classes)



# Convert labels to categorical (one-hot encoding)
num_classes = len(set(train_labels))  # Get the number of classes



