import imageio as iio
import numpy as np
import tensorflow as tf
import os

#np.set_printoptions(threshold=np.inf)


def get_examples(images_path, labels_path):
    """
    Returns tuple of (images, labels)
    Where each image is of size 100x100
    and each label is 100x100x2 (one-hot encoded, "is this pixel a stream or no?")

    :param images_path: path to DIRECTORY where images are kept
    :param labels_path: see above but for labels
    """

    images_lst = []

    # https://newbedev.com/how-to-iterate-over-files-in-a-given-directory
    for filename in os.listdir(images_path):
        if filename.endswith(".png"):
            img_with_chan = np.reshape(iio.imread(images_path + filename), (100, 100, 1))
            images_lst.append(img_with_chan)

    # print(images_lst[0].shape)
    #print(images_lst[0])


    labels_lst = []

    for filename in os.listdir(labels_path):
        if filename.endswith(".png"):
            labels_lst.append(iio.imread(labels_path + filename))

    # print(labels_lst[2].shape)

    # print("BEFORE")
    # for r in labels_lst[10]:
    #     print(r)

    # relabel the labels using fourth channel
    # hacky, couldn't figure a better way to do this in numpy
    # even though one def exists
    encoded_labels=[np.zeros((100, 100)) for l in labels_lst]

    for i in range(len(labels_lst)):
        l = labels_lst[i]
        e = encoded_labels[i]
        for x in range(l.shape[0]):
            for y in range(l.shape[1]):
                if l[x][y][3] == 0:
                    e[x][y] = 0
                else:
                    e[x][y] = 1

    # print("AFTER")
    # print(encoded_labels[10].shape)
    # for r in encoded_labels[10]:
    #     print(r)         
    
    
    return images_lst, encoded_labels



