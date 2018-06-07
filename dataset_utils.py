import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from scipy.ndimage import imread
from keras.utils.np_utils import to_categorical


STANFORD_PATH = "dataset/stanfordBackground"

def getImagesAndLabels(pref="training"):
    imagePath = "%s/%sImages/" % (STANFORD_PATH, pref)
    labelPath = "%s/%sLabels/" % (STANFORD_PATH, pref)
    imagefiles = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]
    labelfiles = ["%s.regions.txt" % s.split('.')[0] for s in imagefiles]
    imagefiles = ["%s%s" % (imagePath, f) for f in imagefiles]
    labelfiles = ["%s%s" % (labelPath, f) for f in labelfiles]
    
    images = []
#     totalFiles = 10
    totalFiles = len(imagefiles)
    validShape = (240, 320, 3)
    i_h, i_w, i_c = validShape
    labels = []
    for i in range(totalFiles):
        im = imread(imagefiles[i])
        label = np.loadtxt(labelfiles[i], delimiter=' ')
        h,w = im.shape[0], im.shape[1]
        lh, lw = label.shape
        assert lh == h and lw == w
        if im.shape != validShape:
            if h*w >= validShape[0]*validShape[1]:
                im = np.resize(im, validShape)
                label = np.resize(label, (validShape[0], validShape[1]))
            else:
                continue
        images.append(im)
        labels.append(label)
        
    images = np.stack(images)
    mean_pixel = images.mean(axis=(1, 2, 3), keepdims=True)
    std_pixel = images.std(axis=(1, 2, 3), keepdims=True)
    images = (images - mean_pixel)/std_pixel
    labels = [to_categorical(np.reshape(l, (240*320)), 8) for l in labels]
    labels = np.stack(labels, axis=0)
#     images = np.split(images, images.shape[0], axis=0)
    return images, labels

class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))
    
def ConstructDataset(pref="training"):
    X, y = getImagesAndLabels(pref)
    train_dset = Dataset(X, y, batch_size=50, shuffle=True)
    return train_dset

# def _parseImage(imageFileName, label):
#     image_decode = tf.image.decode_image(tf.read_file(imageFileName))
#     return image_decode, label

# def _createDatasetIterator(dims, N, batch_size, epochs):
#     features_placeholder = tf.placeholder(tf.string, N)
#     labels_placeholder = tf.placeholder(tf.float, dims)
#     dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
#     datset = dataset.map(_parseImage)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat(epochs)
#     iterator = dataset.make_initializable_iterator()
#     return iterator

# def getStanfordTrainingData():    
#     path = "dataset/stanfordBackground"
#     imagePath = "%s/trainingImages/" % path
#     labelPath = "%s/trainingLabels/" % path
#     imagefiles = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]
#     labelfiles = ["%s.regions.txt" % s.split('.')[0] for s in imagefiles]
#     imagefiles = ["%s%s" % (imagePath, f) for f in imagefiles]
#     labelfiles = ["%s%s" % (labelPath, f) for f in labelfiles]
    
#     labels = _readLabels(labelfiles)
#     dims = labels[0].shape
#     N = len(labels)
#     iterator = _createDatasetIterator(dims, N)
#     return imagefiles, labels, iterator
# getImagesAndLabels()