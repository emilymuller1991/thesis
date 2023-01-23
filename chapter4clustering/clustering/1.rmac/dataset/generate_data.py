import tensorflow as tf
import os
import importlib
import numpy as np


# this function reads in images stored loosely in a folder given by input dataset. 
# subset splits into training and validation
# batch size can be changed to return the entire data set

def generate_data(dataset, img_dim, batch_size, subset, shuffle=True):
    # Augmentation strategy
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1./255,
        featurewise_center=True,
        validation_split=0.2)

    # generator for training data
    train_generator = train_datagen.flow_from_directory(
        dataset,
        target_size=(img_dim, img_dim),
        batch_size=batch_size,
        shuffle = shuffle,
        subset=subset[0])
    # can this be shortened by just obtaining filenames?
    files = []    
    for file in train_generator.filenames:
        files.append(file)
    # generator for test data
    test_generator = train_datagen.flow_from_directory(
        dataset,
        target_size=(img_dim, img_dim),
        batch_size=batch_size,
        subset=subset[1])

    # create tf dataset iterator
    ds = tf.data.Dataset.from_generator(lambda: train_generator,
                        output_types=(tf.float32, tf.float32),
                        output_shapes=([None, img_dim, img_dim, 3], [None,1])
                        )
    # create tf dataset iterator for test with batch size equal to validation_split * # .png's in folder.
    ds_test = tf.data.Dataset.from_generator(lambda: test_generator,
                    output_types=(tf.float32, tf.float32),
                    output_shapes=([None, img_dim, img_dim, 3], [None,1])
                    )

    # require test_unbatched for clustering too
    # train_unbatched = 0 # for clustering on the entire dataset - needs it's own computational work around
    # train_labels = 0
    # test_labels = 0
    # classes = 0
    img_dim = (img_dim, img_dim, 3)

    return ds, ds_test, img_dim, files