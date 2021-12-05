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

    # relabel the labels using fourth channel
    encoded_labels=[np.zeros((100, 100)) for l in labels_lst]

    for i in range(len(labels_lst)):
        l = labels_lst[i]
        l_4 = l[:, :, 3]
        encoded_labels[i] = np.where(l_4 == 0, 0.0, 1.0)

    return images_lst, encoded_labels



