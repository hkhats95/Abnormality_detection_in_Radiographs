import numpy as np
import cv2
from keras.utils import to_categorical
from random import randint

# image size
IMG_W = 320
IMG_H = 320
IMG_C = 3

def train_batch_img_paths_class_generator(img_paths, img_class, batch_size):
    # generating batches as many times requested
    while True:
        cnt = len(img_paths)
        permute = np.random.permutation(cnt).astype(int)
        permuted_img_paths = img_paths[permute]
        permuted_img_class = img_class[permute]
        i = 0
        while i < cnt:
            if (i + batch_size) <= cnt:
                batch_img_paths = permuted_img_paths[i:i + batch_size]
                batch_img_class = permuted_img_class[i:i + batch_size]
                i += batch_size
                yield batch_img_paths, batch_img_class
            else:
                batch_img_paths = permuted_img_paths[i:]
                batch_img_class = permuted_img_class[i:]
                i += batch_size
                yield batch_img_paths, batch_img_class
                
def valid_batch_img_paths_class_generator(img_paths, img_class, batch_size):
    # generating batches as many times requested
    while True:
        cnt = len(img_paths)
        permute = np.random.permutation(cnt).astype(int)
        permuted_img_paths = img_paths[permute]
        permuted_img_class = img_class[permute]
        i = 0
        while i < cnt:
            if (i + batch_size) <= cnt:
                batch_img_paths = permuted_img_paths[i:i + batch_size]
                batch_img_class = permuted_img_class[i:i + batch_size]
                i += batch_size
                yield batch_img_paths, batch_img_class
            else:
                batch_img_paths = permuted_img_paths[i:]
                batch_img_class = permuted_img_class[i:]
                i += batch_size
                yield batch_img_paths, batch_img_class

def img_preprocessing(img, flip=True):
    img = cv2.resize(img, (IMG_H, IMG_W), interpolation = cv2.INTER_AREA) # resize to 320 X 320
    img = (img - np.mean(img))/np.std(img) # normalization
    if randint(0, 1) and flip:
        img = cv2.flip(img, 1) # horizontal flip
#     if randint(0, 1) and flip:
#         img = cv2.flip(img, 0) # vertical flip
    return img


def train_generator(img_paths, img_class, batch_size, NUM_CLASS, flip=True):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths, batch_img_class in train_batch_img_paths_class_generator(img_paths, img_class, batch_size):
            num_img = len(batch_img_paths)
            batch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, flip)
                batch_x[i] = img
            batch_y = to_categorical(batch_img_class, num_classes=NUM_CLASS)
            yield (batch_x, batch_y)
            
def validation_generator(img_paths, img_class, batch_size, NUM_CLASS, flip=True):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths, batch_img_class in valid_batch_img_paths_class_generator(img_paths, img_class, batch_size):
            num_img = len(batch_img_paths)
            vbatch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, flip)
                vbatch_x[i] = img
            vbatch_y = to_categorical(batch_img_class, num_classes=NUM_CLASS)
            yield (vbatch_x, vbatch_y)
            
def predic_batch_img_paths_generator(img_paths, batch_size):
    cnt = len(img_paths)
    i = 0
    while i < cnt:
        if (i + batch_size) <= cnt:
            batch_img_paths = img_paths[i:i + batch_size]
            i += batch_size
            yield batch_img_paths
        else:
            batch_img_paths = img_paths[i:]
            i += batch_size
            yield batch_img_paths
            
def prediction_generator(img_paths, batch_size):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths in predic_batch_img_paths_generator(img_paths, batch_size):
            num_img = len(batch_img_paths)
            pbatch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, False)
                pbatch_x[i] = img
            yield pbatch_x

# print("success")