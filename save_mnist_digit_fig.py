# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def draw_digit(data):
    size = int(len(data)**.5)
    Z = data.reshape(size,size)   # convert from vector to matrix
    plt.imshow(Z,interpolation="None")
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

def draw_digits(data, fname="fig.png"):
    for i in xrange(3*3):
        plt.subplot(331+i)
        draw_digit(data[i])
    plt.savefig(fname)

def save_images(data_list,filename,shape="auto"):
    n_data = len(data_list)
    if shape == "auto":
        sqr = int(n_data**.5)
        if sqr*sqr != n_data:
            shape = (sqr+1,sqr+1)
        else:
            shape = (sqr,sqr)
    plt.figure(figsize=(sqr,sqr))
    for i,data in enumerate(data_list):
        plt.subplot(shape[0],shape[1],i+1)
        plt.gray()
        size = int(len(data)**.5)
        Z = data.reshape(size,size)
        plt.imshow(Z,interpolation='nearest')
        plt.tick_params(labelleft='off',labelbottom="off")
        plt.tick_params(axis='both',which='both',left='off',bottom='off',
                       right='off',top='off')
        plt.subplots_adjust(hspace=.05)
        plt.subplots_adjust(wspace=.05)
    plt.savefig(filename)
    print "save figure:{}",filename

def plot_images(data_list,data_shape="auto",fig_shape="auto"):
    """
    plotting data on current plt object.
    In default,data_shape and fig_shape are auto.
    It means considered the data as a sqare structure.
    """
    n_data = len(data_list)
    if data_shape == "auto":
        sqr = int(n_data**.5)
        if sqr*sqr != n_data:
            data_shape = (sqr+1,sqr+1)
        else:
            data_shape = (sqr,sqr)
    plt.figure(figsize=data_shape)

    for i,data in enumerate(data_list):
        plt.subplot(data_shape[0],data_shape[1],i+1)
        plt.gray()
        if fig_shape == "auto":
            fig_size = int(len(data)**.5)
            if fig_size **2 != len(data):
                fig_shape = (fig_size+1,fig_size+1)
            else:
                fig_shape = (fig_size,fig_size)
        Z = data.reshape(fig_shape[0],fig_shape[1])
        plt.imshow(Z,interpolation='nearest')
        plt.tick_params(labelleft='off',labelbottom="off")
        plt.tick_params(axis='both',which='both',left='off',bottom='off',
                       right='off',top='off')
        plt.subplots_adjust(hspace=.05)
        plt.subplots_adjust(wspace=.05)
