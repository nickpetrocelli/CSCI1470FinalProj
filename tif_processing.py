from tifffile import imread
import os
import numpy as np
from PIL import Image

Path_map = '../../data/worldview/'
Path_river = '../../data/stream_networks/'
map_file_names = ["stream_network_1","stream_network_2"]
river_file_names = ["river_label_1","river_label_2"]

for y in range(1):
    map_im = imread(Path_map + map_file_names[y] + '.tif')
    river_im = imread(Path_river + river_file_names[y] + '.tif')

    map_imgwidth = map_im.shape[1]
    map_imgheight = map_im.shape[0]

    height = 100
    width = 100

    k = 0

    map_store_path = 'C:/Users/sarah/Documents/cs1470/FinalProj/data/sliced_images/' + map_file_names[y] + '/'
    label_store_path = 'C:/Users/sarah/Documents/cs1470/FinalProj/data/sliced_images/' + river_file_names[y] + '/'

    for i in range(0, map_imgheight, height):
        for j in range(0, map_imgwidth, width):
            a = map_im[i:i+height, j:j+width]
            b = river_im[i:i+height, j:j+width]

            # Only keep images with >=60% non blank pixels
            ratio = 1 - len(a[a==0])/(a.shape[0]*a.shape[1])

            if ratio >= 0.6 and a.shape[0] == height and a.shape[1] == width:
                # Save map images
                map_data = Image.fromarray(a)
                map_data.save(os.path.join(map_store_path,"IMG-%s.png" % k))
                # Save label images
                river_data = Image.fromarray(b)
                river_data.save(os.path.join(label_store_path,"IMG-%s.png" % k))

                k +=1

    print("Finished slicing network ", y)
