import os
import pickle
import io
import time
import bson
import threading

import numpy as np
import pandas as pd
import keras
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_v3 import InvepctionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def get_features_label(documents, batch_size=32, return_labels=True):
    for batch in grouper(batch_size, documents):
        images = []
        labels = []
        
        for document in batch:
            category = document.get('category_id', '')
            img = document.get('imgs')[0]
            data = io.BytesIO(img.get('picture', None))
            im = imread(data)

            if category:
                label = labelencoder.transform([category])
            else:
                label = None

            im = im.astype('float32') / 255.0

            images.append(im)
            labels.append(label)

        if return_labels:
            yield np.array(images), np.array(labels)
        else:
            yield np.array(images)


