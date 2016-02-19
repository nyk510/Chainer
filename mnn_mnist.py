# -*- coding: utf-8 -*-
# MnsitDataの多層NN
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
from save_mnist_digit_fig import save_images,plot_images
from fetch_mnist import * #mnist_data の取得
import pandas as pd

class MNN(Chain):
    def __init__(self,activate = F.relu,dropout=.5):
        super(MNN,self).__init__(
            l1 = L.Linear(28*28,28*28),
            l2 = L.Linear(28*28,100),
            l3 = L.Linear(100,10))
        self.activate=activate
        self.dropout = dropout

    def __call__(self,x,train=True):
        self.h1 = F.dropout(self.activate(self.l1(x)),train=train,ratio=self.dropout)
        self.h2 = F.dropout(self.activate(self.l2(self.h1)),train=train,ratio=self.dropout)
        self.h3 = F.dropout(self.activate(self.l3(self.h2)),train=train,ratio=self.dropout)
        return self.h3

class Train_ChainMoel:
    def __init__(self,model,data,target,dropout=None,corruption_ratio=None):
        self.model = model
        self.data =data
        self.target = target
        self.dropout = dropout
        self.corruption_ratio = corruption_ratio
        self.sum_epochs = 0
        self.batchsize = 100
        self.N = len(data[0])
        self.empirical_loss = []
        self.predictive_loss = []

    def cal_loss(self,train=True):
        i = 0
        if train:
            i = 0
        else:
            i = 1
        print "calculate_loss"
        y = self.model(Variable(self.data[i]),train=False)
        t = Variable(self.target[i])
        sum_loss = F.softmax_cross_entropy(y,t).data
        acc = F.accuracy(y,t).data
        print sum_loss,acc
        return sum_loss,acc

    def setpu_optimizer(self,optimizer=optimizers.AdaDelta):
        self.optimizer = optimizer()
        self.optimizer.setup(self.model)

    def start(self,n_epoch=5):

        self.empirical_loss.append(self.cal_loss())
        self.predictive_loss.append(self.cal_loss(train=False))

        for epoch in range(1,n_epoch+1):
            print "epoch",epoch
            perm = np.random.permutation(self.N)

            for i in range(0,self.N,self.batchsize):
                x = Variable(data[0][perm[i:i+self.batchsize]])
                t = Variable(target[0][perm[i:i+self.batchsize]])
                self.model.zerograds()
                y = self.model(x)
                loss = F.softmax_cross_entropy(y,t)
                loss.backward()
                self.optimizer.update()

            self.empirical_loss.append(self.cal_loss())
            self.predictive_loss.append(self.cal_loss(train=False))

    def plot_pred_and_emp_graph(self):
        empirical_loss = np.array(self.empirical_loss)
        predictive_loss = np.array(self.predictive_loss)
        plt.subplot(1,2,1)
        plt.plot(empirical_loss[:,0],"o-",label="emp_loss")
        plt.plot(predictive_loss[:,0],"o-",label="pred_loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(empirical_loss[:,1],"o-",label="emp_acc")
        plt.plot(predictive_loss[:,1],"o-",label="pred_acc")
        plt.ylim(0.9,1.0)
        plt.legend()
        plt.show()

if __name__=="__main__":
    data,target = fetch_mnist()
    train_mnn = Train_ChainMoel(model=MNN(),data=data,target=target,dropout=.1)
    train_mnn.setpu_optimizer()
    train_mnn.start(n_epoch=100)
    train_mnn.plot_pred_and_emp_graph()
    with open("mnn_mnist_log.txt",mode='w') as wf:
        wf.write('epoch\tempirical_loss\tempirical_acc\tpredictive_loss\tpredictive_acc\n')
        for i in range(len(train_mnn.empirical_loss)):
            wf.write(
            str(i)+'\t'
            +str(train_mnn.empirical_loss[i][0])+'\t'
            +str(train_mnn.empirical_loss[i][1])+'\t'
            +str(train_mnn.predictive_loss[i][0])+'\t'
            +str(train_mnn.predictive_loss[i][1])+'\n')
    serializers.save_npz('mnn.model',train_mnn.model)
