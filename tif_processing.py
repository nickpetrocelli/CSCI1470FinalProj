from tifffile import imread
import os
import numpy as np
from PIL import Image

Path = '../../data/worldview/'
file_names = ["stream_network_1_buff_50m", "stream_network_1_buff_100m", "stream_network_2_buff_50m", "stream_network_2_buff_100m"]

for y in range(4):
    im = imread(Path + file_names[y] + '.tif')

    imgwidth = im.shape[1]
    imgheight = im.shape[0]

    height = 100
    width = 100

    k = 0

    #store_path = '../../data/worldview/sliced_images/' + file_names[y] + '/'
    store_path = 'C:/Users/sarah/Documents/cs1470/FinalProj/data/sliced_images/' + file_names[y] + '/'
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            a = im[i:i+height, j:j+width]

            # Only keep images with >=60% non blank pixels
            ratio = 1 - len(a[a==0])/(a.shape[0]*a.shape[1])

            if ratio >= 0.6 and a.shape[0] == height and a.shape[1] == width:
                data = Image.fromarray(a)
                data.save(os.path.join(store_path,"IMG-%s.png" % k))
                k +=1

    print("Finished slicing tifs from group ", y)
