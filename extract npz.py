from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

import numpy as np
import os
import glob

from keras.utils import np_utils
from keras_preprocessing.image import load_img, img_to_array
from sklearn.datasets import load_files
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def extract_Resnet50(file_paths):
    tensors = paths_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_resnet50(tensors)
    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('catImages/train')
test_files, valid_targets = load_dataset('catImages/test')
valid_files, test_targets = load_dataset('catImages/valid')


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

train_resnet50 = extract_Resnet50(train_files)
valid_resnet50 = extract_Resnet50(valid_files)
test_resnet50 = extract_Resnet50(test_files)
print("Resnet50 shape", train_resnet50.shape[1:])

np.savez('bottleneck_features/CatResnet50Data.npz',train=train_resnet50,valid=valid_resnet50,test=test_resnet50)



