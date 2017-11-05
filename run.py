import os
import pickle
import io
import itertools
import time
import bson
import threading

import numpy as np
import pandas as pd
import keras
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from utils.utils import grouper, threadsafe_iter, threadsafe_generator, get_features_label

def create_model(num_classes=None):

    # Pre-trained base model (InceptionV3) 
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model (InceptionV3) layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def main():

    if os.path.isfile('labelencoder.pkl'):
        with open('labelencoder.pkl', 'rb') as f:
            labelencoder = pickle.load(f)
        categories = pd.read_csv('categories.csv')
        
    else:
        documents = bson.decode_file_iter(open('data/train.bson', 'rb'))
        categories = [(d['_id'], d['category_id']) for d in documents]
        categories = pd.DataFrame(categories, columns=['id', 'cat'])

        labelencoder = LabelEncoder()
        labelencoder.fit(categories.cat.unique().ravel())

        with open('labelencoder.pkl', 'wb') as f:
            pickle.dump(labelencoder, f)

        categories.to_csv('categories.csv')

    try:
        inception = keras.models.load_model('inceptionv3-finetune.h5')
    except:
        inception = create_model(num_classes=len(labelencoder.classes_))

    callback = keras.callbacks.TensorBoard(
                log_dir='./logs/inception/2/{}'.format(time.time())
            )

    generator = get_features_label(bson.decode_file_iter(open(
        'data/train.bson', 'rb')))

    inception.fit_generator(
        generator=generator,
        epochs=300,
        steps_per_epoch=500,
        callbacks=[callback],
        validation_data=generator,
        validation_steps=50
    )

    inception.save('inceptionv3-finetune.h5')

    for layer in inception.layers[:249]:
        layer.trainable = False
    for layer in inception.layers[249:]:
        layer.trainable = True

    from keras.optimizers import SGD
    inception.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    
    callback = keras.callbacks.TensorBoard(
            log_dir='./logs/inception/{}'.format(time.time())
            )

    generator = get_features_label(bson.decode_file_iter(open('data/train.bson', 'rb')))

    inception.fit_generator(
        generator=generator,
        epochs=300,
        steps_per_epoch=500,
        callbacks=[callback],
        validation_data=generator,
        validation_steps=50
    )

    inception.save('inceptionv3-finetune-2.h5')



if __name__=="__main__":
    main()
