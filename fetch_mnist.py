# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def fetch_mnist():
    """
    return (data,target)
    data and target: [train,test]
    """
    print '--- fetch MNIST dataset ---'
    mnist = fetch_mldata('MNIST original')
    mnist.data   = mnist.data.astype(np.float32)
    mnist.data  /= 255
    mnist.target = mnist.target.astype(np.int32)

    data_train,\
    data_test,\
    target_train,\
    target_test = train_test_split(mnist.data, mnist.target)

    data = [data_train, data_test]
    target = [target_train, target_test]
    return data,target
