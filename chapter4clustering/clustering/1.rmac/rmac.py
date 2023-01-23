from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.layers import Lambda, Dense, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import numpy as np
import utils

from dataset import generate_data
import time
import pandas as pd
import re
import os

from PIL import Image

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data', 'mnist', 'dataset')
flags.DEFINE_float('start', 0.0, 'starting file')
flags.DEFINE_float('end',  1.0, 'end_file')
flags.DEFINE_string('prefix', '/rds/general/user/emuller/home/emily/phd/', 'when using RDS')

start = np.float(FLAGS.start)
end = np.float(FLAGS.end)


def addition(x):
    sum = K.sum(x, axis=1)
    return sum


def weighting(input):
    x = input[0]
    w = input[1]
    w = K.repeat_elements(w, 512, axis=-1)
    out = x * w
    return out


def rmac(input_shape, num_rois):

    # Load VGG16
    model = tf.keras.applications.vgg16.VGG16(include_top = False, input_shape=input_shape) 
    #img_input = Input(shape=input_shape)
    vgg16_model = tf.keras.models.Model(inputs=model.input, outputs=model.output)
    #vgg16_model = VGG16(utils.DATA_DIR + utils.WEIGHTS_FILE, input_shape)

    # Regions as input
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    # ROI pooling
    x = RoiPooling([1], num_rois)([vgg16_model.layers[-1].output, in_roi])

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)

    # PCA
    x = TimeDistributed(Dense(512, name='pca',
                              kernel_initializer='identity',
                              bias_initializer='zeros'))(x)

    # Normalization
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x)

    # Addition
    rmac = Lambda(addition, output_shape=(512,), name='rmac')(x)

    # # Normalization
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac)

    # Define model
    model = Model([vgg16_model.input, in_roi], rmac_norm)
    print (model.summary())
    # Load PCA weights
    mat = scipy.io.loadmat(FLAGS.prefix + '003_image_matching/keras_rmac-master/' + utils.DATA_DIR + utils.PCA_FILE)
    b = np.squeeze(mat['bias'], axis=1)
    w = np.transpose(mat['weights'])
    model.layers[-4].set_weights([w, b])

    return model

def preprocess(img):
    (left, upper, right, lower) = (137, 132, 505, 500)
    # Here the image "im" is cropped and assigned to new variable im_crop
    img_crop = img.crop((left, upper, right, lower))
    img_crop = img_crop.resize((640,640),Image.BICUBIC)
    return img_crop
# if __name__ == "__main__":

    # # Load sample image
    # # file = utils.DATA_DIR + 'sample.jpg'
    # img = image.load_img(file)

#     # Resize
    # scale = utils.IMG_SIZE / max(img.size)
    # new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    # print('Original size: %s, Resized image: %s' %(str(img.size), str(new_size)))


    # img = img.resize(new_size)

    # # Mean substraction
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
#     x = utils.preprocess_image(x)

#     # Load RMAC model
#     Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
#     regions = rmac_regions(Wmap, Hmap, 3)
#     print('Loading RMAC model...')
#     model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))

#     # Compute RMAC vector
#     print('Extracting RMAC from image...')
#     RMAC = model.predict([x, np.expand_dims(regions, axis=0)])
#     print('RMAC size: %s' % RMAC.shape[1])
#     print('Done!')
path = FLAGS.data
files = os.listdir(path)
n = len(files)
corrupts = []

start_time = time.time()
if __name__ == "__main__":

    # Load sample image
    #dataset = FLAGS.data
    
    img_dim = 640
    K.set_image_data_format('channels_last')
    #ds, ds_test, img_dim_, files = generate_data(dataset, img_dim, 1, subset=(None, None), shuffle=False)
    K.set_image_data_format('channels_first')
    encoded_images = {}
    counter = 0
    for i in range(int(np.floor(n*start)), int(np.ceil(n*end))):
    #for img in ds:
        try:
            # for running with PIL
            img = Image.open(path + '/' + files[i])
            img = preprocess(img)
            name = files[i][:-4]
            img = np.array(img)
            img = np.expand_dims(img, axis=0) 
            img = tf.transpose(img, [0, 3, 1, 2]).numpy()
            img = img.astype('float32') 
            x = utils.preprocess_image(img)

            # Load RMAC model
            if counter == 0:
                Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
                regions = rmac_regions(Wmap, Hmap, 3)
                model = rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
            else:
                pass

            # Compute RMAC vector
            RMAC = model.predict([x, np.expand_dims(regions, axis=0)])

            encoded_images[name] = [RMAC]
        except:
            corrupts.append(files[i])
        counter += 1
        if counter % 10000 == 0:
            encoded = pd.DataFrame(encoded_images).T
            encoded.to_csv(FLAGS.prefix + '003_image_matching/keras_rmac-master/census2021outputs/census2011_zoom_rmac_feature_vector_zoom_%s.csv' % str(FLAGS.end))
            print('Completed image %s in %s seconds!' % (str(counter) + '_' + name, str(time.time() - start_time) ))
            print ('There are %s corrupt files' %len(corrupts))

encoded = pd.DataFrame(encoded_images).T
encoded.to_csv(FLAGS.prefix + '003_image_matching/keras_rmac-master/census2021outputs/census2011_zoom_rmac_feature_vector_zoom_%s.csv' % str(FLAGS.end))
print (corrupts)
