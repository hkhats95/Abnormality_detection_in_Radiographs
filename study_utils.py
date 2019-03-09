import numpy as np
import cv2
from keras.utils import to_categorical
from random import randint

# image size
IMG_W = 320
IMG_H = 320
IMG_C = 3

def train_batch_img_paths_studies_embeds_generator(img_paths, img_studies, img_embeds, batch_size):
    # generating batches as many times requested
    while True:
        cnt = len(img_paths)
        permute = np.random.permutation(cnt).astype(int)
        permuted_img_paths = img_paths[permute]
        permuted_img_studies = img_studies[permute]
        permuted_img_embeds = img_embeds[permute]
        i = 0
        while i < cnt:
            if (i + batch_size) <= cnt:
                batch_img_paths = permuted_img_paths[i:i + batch_size]
                batch_img_studies = permuted_img_studies[i:i + batch_size]
                batch_img_embeds = permuted_img_embeds[i:i + batch_size]
                i += batch_size
                yield batch_img_paths, batch_img_studies, batch_img_embeds
            else:
                batch_img_paths = permuted_img_paths[i:]
                batch_img_studies = permuted_img_studies[i:]
                batch_img_embeds = permuted_img_embeds[i:]
                i += batch_size
                yield batch_img_paths, batch_img_studies, batch_img_embeds
                
def valid_batch_img_paths_studies_embeds_generator(img_paths, img_studies, img_embeds, batch_size):
    # generating batches as many times requested
    while True:
        cnt = len(img_paths)
        permute = np.random.permutation(cnt).astype(int)
        permuted_img_paths = img_paths[permute]
        permuted_img_studies = img_studies[permute]
        permuted_img_embeds = img_embeds[permute]
        i = 0
        while i < cnt:
            if (i + batch_size) <= cnt:
                batch_img_paths = permuted_img_paths[i:i + batch_size]
                batch_img_studies = permuted_img_studies[i:i + batch_size]
                batch_img_embeds = permuted_img_embeds[i:i + batch_size]
                i += batch_size
                yield batch_img_paths, batch_img_studies, batch_img_embeds
            else:
                batch_img_paths = permuted_img_paths[i:]
                batch_img_studies = permuted_img_studies[i:]
                batch_img_embeds = permuted_img_embeds[i:]
                i += batch_size
                yield batch_img_paths, batch_img_studies, batch_img_embeds

def img_preprocessing(img, flip=True):
    img = cv2.resize(img, (IMG_H, IMG_W), interpolation = cv2.INTER_AREA) # resize to 320 X 320
    img = (img - np.mean(img))/np.std(img) # normalization
#     if randint(0, 1) and flip:
#         img = cv2.flip(img, 1) # horizontal flip
#     if randint(0, 1) and flip:
#         img = cv2.flip(img, 0) # vertical flip
    return img


def train_generator(img_paths, img_studies, img_embeds, batch_size, flip=True):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths, batch_img_studies, batch_img_embeds in train_batch_img_paths_studies_embeds_generator(img_paths, img_studies, img_embeds, batch_size):
            num_img = len(batch_img_paths)
            batch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, flip)
                batch_x[i] = img
            batch_y = batch_img_studies
            yield ([batch_x, batch_img_embeds], batch_y)
            
def validation_generator(img_paths, img_studies, img_embeds, batch_size, flip=True):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths, batch_img_studies, batch_img_embeds in valid_batch_img_paths_studies_embeds_generator(img_paths, img_studies, img_embeds, batch_size):
            num_img = len(batch_img_paths)
            vbatch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, flip)
                vbatch_x[i] = img
            vbatch_y = batch_img_studies
            yield ([vbatch_x, batch_img_embeds], vbatch_y)
            
def predic_batch_img_paths_embeds_generator(img_paths, img_embeds, batch_size):
    cnt = len(img_paths)
    i = 0
    while i < cnt:
        if (i + batch_size) <= cnt:
            batch_img_paths = img_paths[i:i + batch_size]
            batch_img_embeds = img_embeds[i:i + batch_size]
            i += batch_size
            yield batch_img_paths, batch_img_embeds
        else:
            batch_img_paths = img_paths[i:]
            batch_img_embeds = img_embeds[i:]
            i += batch_size
            yield batch_img_paths, batch_img_embeds
            
def prediction_generator(img_paths, img_embeds, batch_size):
    while True: # for generating batch as many times required for training the model
        for batch_img_paths, batch_img_embeds in predic_batch_img_paths_embeds_generator(img_paths, img_embeds, batch_size):
            num_img = len(batch_img_paths)
            pbatch_x = np.zeros((num_img, IMG_H, IMG_W, IMG_C))
            for i in range(num_img):
                img = cv2.imread(batch_img_paths[i].strip())
                img = img_preprocessing(img, False)
                pbatch_x[i] = img
            yield [pbatch_x, batch_img_embeds]

# print("success")