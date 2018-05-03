import numpy as np
from numpy import linalg
import gzip
import cv2
import os

def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        #print magic, num_images, rows, cols
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

def loadmnist():
    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')
    return train_x, train_y, test_x, test_y

# read and normalize catech data size and apply hog to this datadset for feature selection
def readcatech():
    folder_path = "./data/Catech10/"
    category = 0
    dataset=[]
    label = []
    for foldername in os.listdir(folder_path):
        category_path = folder_path + foldername + "/"
        for filename in os.listdir(category_path):
            image_path=category_path + filename
            image = cv2.imread(image_path)
            image = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
            hog=cv2.HOGDescriptor()
            image=hog.compute(image)
            image = np.reshape(image,(-1))
            label.append(category)
            dataset.append(image)
        category =category +1
    dataset = np.array(dataset)
    label = np.array(label)
    return dataset, label

#since each image has different size, we need do some preproceesing of data
def loadcatech():
    dataset, label = readcatech()
    data = np.zeros((dataset.shape[0],(dataset.shape[1]+1)))
    data[:,0:dataset.shape[1]] = dataset[:,0:dataset.shape[1]]
    data[:,dataset.shape[1]] = label
    randomdata=np.random.permutation(data)
    row = int(randomdata.shape[0]/2)
    train_x = randomdata[0:row,0:(randomdata.shape[1]-1)]
    train_y = randomdata[0:row,randomdata.shape[1]-1]
    test_x  = randomdata[row:randomdata.shape[0],0:(randomdata.shape[1]-1)]
    test_y  = randomdata[row:randomdata.shape[0],randomdata.shape[1]-1]
    return train_x, train_y, test_x, test_y

