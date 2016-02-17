# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as c
import matplotlib.pyplot as plt
from autoencoder import *
from save_mnist_digit_fig import *

class DAE(Chain):
    """
    3 hidden layer deep autoencoder
    """
    def __init__(self,n_inputs=28*28,n_hidden_layers=[28*28,100],
                     corruption_ratio=.0,
                     activate = F.sigmoid,
                     dropout_ratio=.1):
        super(DAE,self).__init__(
            l1=L.Linear(28*28,n_hidden_layers[0]),
            l2=L.Linear(n_hidden_layers[0],n_hidden_layers[1]),
            l3=L.Linear(n_hidden_layers[1],n_hidden_layers[0]),
            l4=L.Linear(n_hidden_layers[0],28*28)
                )
        self.n_inputs = n_inputs
        self.n_hidden_layers = n_hidden_layers
        self.corruption_ratio = corruption_ratio
        self.dropout_ratio = dropout_ratio
        self.activate = activate

    def __call__(self,x,is_training=True):
        """
        dirty :(
        あとでより深層に対応できるように直したい
        """
        h1 = F.dropout(self.activate(self.l1(x)),train=is_training,ratio=self.dropout_ratio)
        h2 = F.dropout(self.activate(self.l2(h1)),train=is_training,ratio=self.dropout_ratio)
        h3 = F.dropout(self.activate(self.l3(h2)),train=is_training,ratio=self.dropout_ratio)
        h4 = F.dropout(self.activate(self.l4(h3)),train=is_training,ratio=self.dropout_ratio)
        return h4

    def to_string(self):
        string = ""
        for i,n_hidden in enumerate(self.n_hidden_layers):
            string += "nhidden"+str(i+1)+"="+str(n_hidden)
        string += "corp=" + str(int(self.corruption_ratio*100.))
        string += "activate="+str(self.activate.__name__)
        return string

class Train_DAE(Train_AutoEncoder):
    def __init__(self,tt_data,model,corruption_ratio=.3,optimizer=optimizers.Adam,rng=np.random.RandomState(71)):
        super(Train_DAE,self).__init__(
            tt_data=tt_data,
            model=model,
            corruption_ratio=corruption_ratio,
            optimizer = optimizer,
            rng = rng
        )

    def pre_training(self,epoch=5,batchsize=100):
        batchsize = batchsize
        self.datasize = len(self.train_data)
        self.mini_ae = []
        print "---start_first_layer----"
        #classifer = Classifer(Autoencoder(self.model[0],self.model[3]))
        #こうやるとgiven link is already registered to another chainとなる
        #どうやらlinkは一つのchainにしか属せないらしい。なのでコピーしないとだめ。
        ae1 = AutoEncoder(n_inputs=self.model.n_inputs,n_hidden=self.model.n_hidden_layers[0])
        train1 = Train_AutoEncoder(model=ae1,tt_data=[self.train_data,self.test_data])
        train1.start(epoch=epoch)
        self.mini_ae.append(train1)
        print "---start_second_layer---"

        train_h1 = ae1(Variable(self.train_data),hidden=True,is_training=False).data
        ae2 = AutoEncoder(n_inputs=self.model.n_hidden_layers[0],n_hidden = self.model.n_hidden_layers[1])
        train2 = Train_AutoEncoder(model=ae2,tt_data=[train_h1,self.test_data])
        train2.start(epoch=epoch)
        self.mini_ae.append(train2)
        print type(ae1)
        print type(ae1.encoder)

        #始めはnetworkをchainlistで書こうと思ってたんだけど
        # chainlistはtupleなので書き換えができない‼️以下はエラー
        # self.model[0] = ae1.copy_encoder()
        # self.model[1] = ae2.copy_encoder()
        # self.model[2] = ae2.copy_decoder()
        # self.model[3] = ae1.copy_encoder()

        #とりあえずchainで名前つけることにした
        self.model.l1 = ae1.copy_encoder()
        self.model.l2 = ae2.copy_encoder()
        self.model.l3 = ae2.copy_decoder()
        self.model.l4 = ae1.copy_decoder()

    def fine_tuning(self,epoch=5,batchsize=100):
        self.epoch = epoch
        self.start(epoch=epoch,batchsize=batchsize)

if __name__ == '__main__':
    print 'fetch MNIST dataset'
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
    n_hidden_layers = [10*10,5*5]

    dae = DAE(n_hidden_layers=n_hidden_layers)
    train_dae = Train_DAE(tt_data=data,model=dae)
    train_dae.pre_training(epoch=20)
    train_dae.fine_tuning(epoch=10)
    condition = train_dae.condition_to_string()
    save_images(train_dae.model.l2.W.data,filename='DAE_'+condition+'layer2.png')
    save_images(train_dae.model.l1.W.data,filename='DAE_'+condition+'layer1.png')
