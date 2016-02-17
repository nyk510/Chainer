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
    sqr = int(n_data**.5)
    plt.figure(figsize=(sqr,sqr))
    for i,data in enumerate(data_list):
        if shape == "auto":
            plt.subplot(sqr,sqr,i+1)
        else:
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
