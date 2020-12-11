#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import time
import Additive_mixing_layers_extraction
from scipy.spatial import ConvexHull
import PIL.Image as Image
import json
import glob
import os
import shutil
import matplotlib.pyplot as plt
Additive_mixing_layers_extraction.DEMO = True
np.set_printoptions(precision=3, suppress=True)


def add_zero(pts):
    """Add zero to set of points if it's not there."""
    zero_index_arr = []
    new_pts = []
    for i, point in enumerate(pts):
        if np.isclose(point, [0, 0, 0]).all():
            zero_index_arr.append(i)
            new_pts.append([0, 0, 0])
        else:
            new_pts.append(point)
    pts = np.array(new_pts)

    if len(zero_index_arr) == 1:
        zero_index = zero_index_arr[0]
    elif len(zero_index_arr) > 1:
        raise ValueError("Multiple zeros in conv hull")
    # otherwise, add it in the last position
    else:
        zero_index = len(pts)
        pts = np.append(pts, [[0, 0, 0]], axis=0)
    return pts, zero_index


def get_snowcone_palette(pts, filepath):
    """Compute the snowcone palette from an existing palette choice."""
    print("original palette is:")
    print(pts)
    print("In full palette, there are {} colors".format(len(pts)))
    # find (0,0,0) point if it's in the data set within some tolerance, but
    # replace it with actual (0, 0, 0)
    pts, zero_index = add_zero(pts)
    M = len(pts)
    # "snowcone hull" only includes vertices on a face of the convex hull that
    # include the origin
    hull = ConvexHull(pts)
    good_indices = np.zeros((M), dtype=bool)
    for simp in hull.simplices:
        if np.isin(zero_index, simp):
            for vert in simp:
                good_indices[vert] = True
    snowcone_hull = pts[good_indices]
    print("snowcone hull is:")
    print(snowcone_hull)
    # use single linkage clustering to find a hull of size k only
    k = 4
    from scipy.cluster.hierarchy import linkage
    Z = linkage(snowcone_hull, "single")
    print("Single linkage matrix is:")
    print(Z)
    size_of_hull = len(snowcone_hull)
    linkage_row = 0
    ind_to_remove = []
    while size_of_hull > k:
        # reduce size using single linkage clustering
        ind1 = Z[linkage_row, 0]
        ind2 = Z[linkage_row, 1]
        print("indices 1 and 2 are")
        print(ind1, ind2)
        if ind1 < len(snowcone_hull) and \
                np.any(snowcone_hull[int(ind1)] != 0):
            # remove ind 1
            ind_to_remove.append(int(ind1))
            size_of_hull -= 1
        elif ind2 < len(snowcone_hull) and \
                np.any(snowcone_hull[int(ind2)] != 0):
            # remove ind 2
            ind_to_remove.append(int(ind2))
            size_of_hull -= 1
        # if neither is an original index, don't do anything
        print("ind_to_remove is", ind_to_remove)
        linkage_row += 1
    ind_to_remove.sort(reverse=True)
    print(ind_to_remove)
    for i in ind_to_remove:
        snowcone_hull = np.delete(snowcone_hull, i, axis=0)
    print("after deleting, snowcone hull is:")
    print(snowcone_hull)

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

    # extend snowcone verts to the boundary of the rgb cube
    for i in range(len(snowcone_hull)):
        pt = snowcone_hull[i]
        # multiply each point by inverse of its max coordinate
        if max(pt) == 0:
            c = 1
        else:
            c = 1/max(pt)
        snowcone_hull[i] = c * pt
    # Plot extended hull in blue
    ax.plot(snowcone_hull.T[0], snowcone_hull.T[1], snowcone_hull.T[2], "bo")

    snowcone_hull_filepath = filepath[:-4]+"-snowcone_plot.pdf"
    fig.savefig(snowcone_hull_filepath)

    return snowcone_hull


def save_weights(img, palette_rgb, mixing_weights, output_prefix):
    # redefine output prefix so that we save the outputs in a new folder
    layers_dir = "./test/layers"
    if os.path.exists(layers_dir):
        shutil.rmtree(layers_dir)
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

    composed_image_filename =\
        layer_output_prefix + "-palette_size-" + str(len(palette_rgb)) +\
        "-composed.png"
    composed_image = np.zeros(img.shape)
    print("mixing weights")
    print(np.max(mixing_weights))
    for i in range(mixing_weights.shape[-1]):
        # print("Creating layer {}".format(i))
        mixing_weights_map_filename =\
            layer_output_prefix + "-palette_size-" + str(len(palette_rgb)) +\
            "-mixing_weights-{}.png".format(palette_rgb[i])
        this_layer_mw = mixing_weights[:, :, i]
        # print("palette_rgb[i] shape:", palette_rgb[i].shape)
        mw = np.repeat(this_layer_mw[:, :, np.newaxis], 3, axis=2)
        Image.fromarray((mw*palette_rgb[i]*255).
                        round().clip(0, 255).
                        astype(np.uint8)).save(mixing_weights_map_filename)
        composed_image += mw*palette_rgb[i]*255
    print("composed_image.shape:", composed_image.shape)
    print("composed image at x=0, y=0")
    print(composed_image[0, 0, :])
    Image.fromarray(composed_image.
                    round().clip(0, 255).
                    astype(np.uint8)).save(composed_image_filename)
    return rmse


def get_bigger_palette_to_show(palette):
    # palette shape is M*3
    c = 50
    palette2 = np.ones((1*c, len(palette)*c, 3))
    for i in range(len(palette)):
        palette2[:, i*c:i*c+c, :] = palette[i, :].reshape((1, 1, -1))
    print("palette2 shape:", palette2.shape)
    return palette2


def main():
    base_dir = "./test/"
    filepath = glob.glob(base_dir + "*.png")[0]
    print(filepath)
    img = np.asfarray(Image.open(filepath).convert('RGB'))/255.0

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
    palette_rgb = get_snowcone_palette(palette_rgb, filepath)
    print("palette_rbg is")
    print(palette_rgb)
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
    mixing_weights = mixing_weights.reshape(
        (img.shape[0], img.shape[1], -1)).clip(0, 1)
    print("Mixing weights 2 shape:", mixing_weights_2.shape)
    print("Mixing weights 1 shape:", mixing_weights_1.reshape((-1, M)).shape)
    print("MW1 at (0,0) is {}".format(mixing_weights_1.reshape((-1, M))[0]))
    print("MW2 at (0,0):")
    print(mixing_weights_2[0])
    print("MW2 at (1,0) or maybe (0,1):")
    print(mixing_weights_2[1])
    print("Mixing weights = MW2 dot MW1 shape:", mixing_weights.shape)
    output_prefix = filepath[:-4]
    pd.DataFrame(mixing_weights_2).to_csv(output_prefix + "_MW2.csv")
    print("class of MW2:", type(mixing_weights_2))
    pd.DataFrame(mixing_weights_1.reshape((-1, M))).to_csv(
        output_prefix + "_MW1.csv")

    print("Mixing weights at x=0, y=0")
    print(mixing_weights[0, 0, :])
    print("sum is", sum(mixing_weights[1, 0, :]))
    print("sum over all:")
    print(np.sum(mixing_weights, axis=(2)))

    end = time.time()
    print("total time: ", end-start)

    output_prefix = filepath[:-4]+'-RGBXY_RGB_black_star_ASAP'
    print("palette rgb")
    print(palette_rgb)
    RMSE = save_weights(arr, palette_rgb, mixing_weights, output_prefix)  # noqa


if __name__ == "__main__":
    main()
