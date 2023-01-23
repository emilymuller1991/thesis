import pickle

DATA_DIR = 'data/'
WEIGHTS_FILE = 'vgg16_weights_th_dim_ordering_th_kernels.h5'
PCA_FILE = 'PCAmatrices.mat'
IMG_SIZE = 1024


def save_obj(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    print ("Object saved to %s." % filename)


def load_obj(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    print ("Object loaded from %s." % filename)
    return obj


def preprocess_image(x):

    # Substract Mean
    x[:, 0, :, :] -= x[:, 0, :, :].mean()
    x[:, 1, :, :] -= x[:, 1, :, :].mean()
    x[:, 2, :, :] -= x[:, 2, :, :].mean()

    # 'RGB'->'BGR'
    x = x[:, ::-1, :, :]

    return x