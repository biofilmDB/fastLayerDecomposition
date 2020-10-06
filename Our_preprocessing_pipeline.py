#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time
import Additive_mixing_layers_extraction
from scipy.spatial import ConvexHull
import PIL.Image as Image
import json
import glob
import os
Additive_mixing_layers_extraction.DEMO = True


def save_weights(img, palette_rgb, mixing_weights, output_prefix):
    # redefine output prefix so that we save the outputs in a new folder
    layers_dir = "./test/layers"
    if not os.path.exists(layers_dir):
        os.makedirs(layers_dir)
    layer_output_prefix = layers_dir + output_prefix.split("test")[1]
    mixing_weights = mixing_weights.reshape(
        (img.shape[0], img.shape[1], -1)).clip(0, 1)
    temp = (mixing_weights.reshape(
        (img.shape[0],
         img.shape[1], -1, 1))*palette_rgb.reshape((1, 1, -1, 3))).sum(axis=2)
    img_diff = temp*255-img*255
    diff = np.square(img_diff.reshape((-1, 3))).sum(axis=-1)
    print('max diff: ', np.sqrt(diff).max())
    print('median diff', np.median(np.sqrt(diff)))
    rmse = np.sqrt(diff.sum()/diff.shape[0])
    print('RMSE: ', np.sqrt(diff.sum()/diff.shape[0]))

    mixing_weights_filename =\
        output_prefix + "-palette_size-" + str(len(palette_rgb)) +\
        "-mixing_weights.js"
    with open(mixing_weights_filename, 'w') as myfile:
        json.dump({'weights': mixing_weights.tolist()}, myfile)

    for i in range(mixing_weights.shape[-1]):
        mixing_weights_map_filename =\
            layer_output_prefix + "-palette_size-" + str(len(palette_rgb)) +\
            "-mixing_weights-%02d.png" % i
        Image.fromarray((mixing_weights[:, :, i]*255).round().clip(0, 255).
                        astype(np.uint8)).save(mixing_weights_map_filename)
    return rmse


def get_bigger_palette_to_show(palette):
    # palette shape is M*3
    c = 50
    palette2 = np.ones((1*c, len(palette)*c, 3))
    for i in range(len(palette)):
        palette2[:, i*c:i*c+c, :] = palette[i, :].reshape((1, 1, -1))
    return palette2


# In[3]:


base_dir = "./test/"

filepaths = glob.glob(base_dir+"*.png")
print(filepaths)


for filepath in filepaths:
    print("#####################")
    print(filepath)
    img = np.asfarray(Image.open(filepath).convert('RGB'))/255.0
    print("img.shape is")
    print(img.shape)
    arr = img.copy()
    X, Y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    XY = np.dstack((X*1.0/img.shape[0], Y*1.0/img.shape[1]))
    data = np.dstack((img, XY))
    # data is a grid of 5d points: r, g, b, x, y
    print("Size of RGBXY data set:")
    print(len(data.reshape((-1, 5))))

    # get palette

    start = time.time()
    # inputs: image, output filename
    palette_rgb = Additive_mixing_layers_extraction.\
        Hull_Simplification_determined_version(
            img, filepath[:-4]+"-convexhull_vertices")
    palette_rgb_255 = palette_rgb*255
    end = time.time()
    M = len(palette_rgb)
    print("palette_rbg = ", palette_rgb)
    print("palette_rgb_255 = ")
    print(palette_rgb_255)
    print("palette size: ", M)
    print("palette extraction time: ", end-start)

    palette_img = get_bigger_palette_to_show(palette_rgb)
    # Image.fromarray((palette_img*255).round().astype(np.uint8)).save(filepath[:-4]+"-convexhull_vertices.png")

    # get layer decomposition

    # for RGBXY RGB black star triangulation.
    start = time.time()
    data_hull = ConvexHull(data.reshape((-1, 5)))
    start2 = time.time()
    print("convexhull on 5D time: ", start2-start)
    mixing_weights_1 = Additive_mixing_layers_extraction.\
        Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates( # noqa
            img.reshape((-1, 3))[data_hull.vertices].reshape((-1, 1, 3)),
            palette_rgb,
            filepath[:-4],
            order=0)
    mixing_weights_2 = Additive_mixing_layers_extraction.\
        recover_ASAP_weights_using_scipy_delaunay(
            data_hull.points[data_hull.vertices],
            data_hull.points, option=3)

    mixing_weights = mixing_weights_2.dot(mixing_weights_1.reshape((-1, M)))
    print("Mixing weights 2 shape:", mixing_weights_2.shape)
    print("Mixing weights 1 shape:", mixing_weights_1.reshape((-1, M)).shape)

    end = time.time()
    print("total time: ", end-start)

    mixing_weights = mixing_weights.reshape(
        (img.shape[0], img.shape[1], -1)).clip(0, 1)

    output_prefix = filepath[:-4]+'-RGBXY_RGB_black_star_ASAP'
    RMSE = save_weights(arr, palette_rgb, mixing_weights, output_prefix)
