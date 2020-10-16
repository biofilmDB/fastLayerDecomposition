#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import numpy as np
import time
import Additive_mixing_layers_extraction
from scipy.spatial import ConvexHull
import PIL.Image as Image
import json
import glob
import os
import matplotlib.pyplot as plt
Additive_mixing_layers_extraction.DEMO = True


def get_snowcone_palette(pts):
    """Compute the snowcone palette from an existing palette choice."""
    M = len(pts)
    print("In full palette, there are {} colors".format(M))
    # add (0, 0, 0) point if needed
    added_zero = False
    if not np.isin(pts, [0, 0, 0]).all(1).any():
        pts = np.append(pts, [[0, 0, 0]], axis=0)
        M += 1
        added_zero = True
    # add (-1, -1, -1) point so we can look from there
    pts = np.append(pts, [[-1, -1, -1]], axis=0)
    qhull_options = "QG" + str(M)
    hull = ConvexHull(pts, qhull_options=qhull_options)
    good_simps = hull.simplices[hull.good]
    good_verts = np.unique(hull.simplices[hull.good])
    good_indices = np.isin(np.arange(M + 1), good_verts)
    if added_zero:
        # should remove 0, which is at M-1 position
        good_indices[M - 1] = False
    snowcone_hull = pts[good_indices]
    # plot the conv hull and the snowcone verts
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Plot all points
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")
    # Plot good verts in green
    ax.plot(snowcone_hull.T[0], snowcone_hull.T[1], snowcone_hull.T[2], "go")

    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")

    conv_hull_filepath = filepath[:-4]+"-convhull_plot.pdf"
    fig.savefig(conv_hull_filepath)

    return snowcone_hull


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
        # print("Creating layer {}".format(i))
        mixing_weights_map_filename =\
            layer_output_prefix + "-palette_size-" + str(len(palette_rgb)) +\
            "-mixing_weights-%02d.png" % i
        this_layer_mw = mixing_weights[:, :, i]
        # print("palette_rgb[i] shape:", palette_rgb[i].shape)
        mw = np.repeat(this_layer_mw[:, :, np.newaxis], 3, axis=2)
        # print("mw shape:", mw.shape)
        Image.fromarray((mw*palette_rgb[i]*255).
                        round().clip(0, 255).
                        astype(np.uint8)).save(mixing_weights_map_filename)
    return rmse


def get_bigger_palette_to_show(palette):
    # palette shape is M*3
    c = 50
    palette2 = np.ones((1*c, len(palette)*c, 3))
    for i in range(len(palette)):
        palette2[:, i*c:i*c+c, :] = palette[i, :].reshape((1, 1, -1))
    print("palette2 shape:", palette2.shape)
    return palette2


def get_sphere_img():
    """Generate a set of points arranged in a sphere for testing algorithms."""
    n = 6
    radius = 0.4
    points = np.zeros([n, n, 3])
    for i in range(n):
        for j in range(n):
            theta = i * math.pi / n
            phi = j * math.pi / n
            points[i, j, 0] = radius * math.sin(phi) * math.cos(theta) + 0.5
            points[i, j, 1] = radius * math.sin(phi) * math.sin(theta) + 0.5
            points[i, j, 2] = radius * math.cos(phi) + 0.5
    return points


# Execute code
# for testing, we'll define a sphere of points instead of reading an image from
# file.
from_file = False

base_dir = "./test/"
if from_file:
    filepath = glob.glob(base_dir + "*.png")[0]
    print(filepath)
    img = np.asfarray(Image.open(filepath).convert('RGB'))/255.0
else:
    filepath = base_dir + "sphere.png"
    img = get_sphere_img()


print("#####################")
print(filepath)
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
palette_rgb = Additive_mixing_layers_extraction.\
    Hull_Simplification_determined_version(
        img,  # input data
        filepath[:-4]+"-convexhull_vertices"  # filepath to write
    )
palette_rgb = get_snowcone_palette(palette_rgb)
end = time.time()
M = len(palette_rgb)
print("snowcone palette size: ", M)
print("palette extraction time: ", end-start)

palette_img = get_bigger_palette_to_show(palette_rgb)
Image.fromarray((palette_img*255).round().astype(np.uint8)).\
    save(filepath[:-4]+"-convexhull_vertices.pdf")

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
