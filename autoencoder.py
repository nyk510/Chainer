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
from save_mnist_digit_fig import *

class AutoEncoder(Chain):
    """
    simple autoencoder
    """
    def __init__(self,n_inputs=784,n_hidden=100,dropout_ratio=.5,activate=F.sigmoid):
        super(AutoEncoder,self).__init__(
            encoder = L.Linear(n_inputs,n_hidden),
            decoder = L.Linear(n_hidden,n_inputs))
        self.dropout_ratio = dropout_ratio
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.activate = activate

    def __call__(self,x,is_training=True,hidden=False):
        h = F.dropout(self.activate(self.encoder(x)),train=is_training,
            ratio = self.dropout_ratio)
        if hidden:
            return h
        y = F.dropout(self.activate(self.decoder(h)),train=is_training,
            ratio = self.dropout_ratio)
        return y

    def to_string(self):
        return "drop="+str(int(self.dropout_ratio*100.))+"activate="+self.activate.__name__+"_nhidden="+str(self.n_hidden)

    def copy_encoder(self):
        initialW = self.encoder.W.data
        initial_bias = self.encoder.b.data
        return F.Linear(self.n_inputs,
            self.n_hidden,
            initialW=initialW,
            initial_bias=initial_bias)

    def copy_decoder(self):
        initialW = self.decoder.W.data
        initial_bias = self.decoder.b.data
        return F.Linear(self.n_hidden,
                        self.n_inputs,
                        initialW=initialW,
                        initial_bias=initial_bias)

class Train_AutoEncoder(object):
    #これがないと継承時にTypeError: must be type, not classobjで怒られる
    def __init__(
            self,
            tt_data,
            model,
            corruption_ratio=.1,
            rng=np.random.RandomState(1),
            optimizer=optimizers.Adam
        ):
        self.train_data,self.test_data = tt_data
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.model = model
        self.corruption_ratio =corruption_ratio
        self.optimizer = optimizer()
        self.rng= rng

    def condition_to_string(self):
        return "alg="+str(self.optimizer.__class__.__name__)+"_epoch="+str(self.epoch)+"_corp="+str(int(self.corruption_ratio*100.))+self.model.to_string()

    def _add_noise(self,x):
        if self.corruption_ratio == .0:
            return x
        mask = self.rng.binomial(size=x.shape,n=1,p=1.-self.corruption_ratio)
        mask = mask.astype(np.float32)
        ret = mask * x
        return ret

    def start(self,epoch=5,batchsize=100):
        self.batchsize = batchsize
        self.epoch = epoch
        classifer = Classifer(predictor=self.model)
        self.optimizer.setup(classifer)
        loss_values = []
        target = self.train_data
        noized_data = self._add_noise(self.train_data)
        print '---start training autoencoder---'
        for n_epoch in range(epoch):
            print "epoch",n_epoch+1
            noized_data = self._add_noise(self.train_data)
            perm = self.rng.permutation(self.train_size)
            for i in xrange(0,self.train_size,batchsize):
                x = Variable(noized_data[perm[i:i+batchsize]])
                t = Variable(target[perm[i:i+batchsize]])
                self.optimizer.update(classifer,x,t)
            loss_values.append(classifer(Variable(self.train_data),Variable(self.train_data)).data)
            print 'loss func:',loss_values[-1]
        self.loss_values = loss_values
        self.classifer = classifer
        return

class Classifer(Chain):
    def __init__(self,predictor):
        super(Classifer,self).__init__(predictor=predictor)
    def __call__(self,x,t):
        y = self.predictor(x)
        self.loss = F.mean_squared_error(y,t)
        return self.loss

if __name__ == '__main__':
    from fetch_mnist import *
    data,target = fetch_mnist()
        # test autoencoder training
    ae = AutoEncoder(
                    n_inputs=784,
                    n_hidden=100,
                    dropout_ratio=.10,
                    activate=F.relu
                )

    train_ae = Train_AutoEncoder(
        tt_data=data,
        model=ae,
        optimizer=optimizers.AdaDelta,
        corruption_ratio=.3)
    train_ae.start(epoch=5)

    filename = train_ae.condition_to_string()
    #save weight figure
    save_images(ae.encoder.W.data,filename=filename+"enc.png")
    save_images(ae.decoder.W.data.T,filename=filename+"dec.png")
