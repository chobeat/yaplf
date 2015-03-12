from ctypes import c_bool

__author__ = 'chobeat'

from yaplf.data import LabeledExample
import unittest
from yaplf.algorithms.svm.classification import *
import random
from numpy import arange,linspace
import warnings
from yaplf.graph import classification_data_plot,MatplotlibPlotter
import yaplf.models.kernel
from yaplf.data import LabeledExample
import matplotlib.pyplot as plt
import pickle
from pylab import *
from os.path import expanduser
import datetime
warnings.simplefilter("error")
import numpy as np

from yaplf.utility.synthdataset import DataGenerator

home = expanduser("~")
thesispath=home+"/git/tesi/Latex/img/"

cf_f=lambda x:("yellow" if x.label==1 else ("blue" if x.label==0 else "red"))
def plot_data(labeled,unlabeled,path):

            dataset=labeled+[LabeledExample(i,0) for i in unlabeled]

            fig=classification_data_plot(dataset,color_function=cf_f)
            fig.savefig(path)



def tmp_plot(alg,labeled,unlabeled,path=None,esvm=True,regrFunc=None,):

            dataset=labeled+[LabeledExample(i,0) for i in unlabeled]
            fig=classification_data_plot(dataset,color_function=cf_f)
            xr=5
            axes = fig.add_subplot(111)

            h=float(2*xr)/len(dataset)
            if alg:
                x=range(-xr,xr)
                y=[-alg.model.decision_function((x_i,0)) for x_i in x]

                xx, yy = np.meshgrid(np.arange(-xr,xr, h),
                         np.arange(-xr, xr, h))
                l=[(x[0],x[1]) for x in np.c_[xx.ravel(), yy.ravel()]]

                Z=np.array([alg.model.decision_function(x) for x in l])
                Z = Z.reshape(xx.shape)

                if esvm:
                     contour_value_eps = [alg.model.tube_radius,-alg.model.tube_radius]

                     contour_style = ('--',) * len(contour_value_eps)
                     plt.contour(xx, yy, Z,contour_value_eps,linestyles=contour_style,colors="g")

                contour_value_eps = [1,-1]
                contour_style = ('-',) * len(contour_value_eps)
                plt.contour(xx, yy, Z,contour_value_eps,linestyles=contour_style,colors="g")
                plt.contour(xx, yy, Z,[0],linewidths=[2],colors="b")
            if regrFunc:
                """
                xx, yy = np.meshgrid(np.arange(-xr,xr, h),
                     np.arange(-3, 3, h))
                l=[[x[0],x[1]] for x in np.c_[xx.ravel(), yy.ravel()]]

                Z=np.array(regrFunc(l))

                Z = Z.reshape(xx.shape)


                contour_style = ('-',) * len(contour_value_eps)

                plt.contour(xx, yy, Z,[0],linestyles=contour_style,colors="green")
                """

                xx=np.arange(-xr,xr, h)
                yy=[regrFunc(x) for x in xx]
                plt.plot(xx,yy,linestyle="-",linewidth=2,color="blue")

            if path:
                fig.savefig(path,bbox_inches='tight')
            else:
               fig.plot()
            plt.close()
if __name__=="__main__":

    """
    #Basic dataset

    p=thesispath+"basicdatalabel.png"
    d=DataGenerator()
    neg=d.generate_from_point([-1,-1],50,1.3,-1)
    pos=d.generate_from_point([1,1],50,1.3,1)
    labeled=neg+pos
    plot_data(labeled,[],p)


    #Mixed dataset

    p=thesispath+"mixeddataset.png"
    d=DataGenerator()
    neg=d.generate_from_point([-1,-1],40,1,-1)
    pos=d.generate_from_point([1,1],40,1,1)
    labeled=neg+pos
    unlabeled=d.generate_from_function(lambda x:-x,50,0.5,-0.8,0.8,0)
    plot_data(labeled,unlabeled,p)

    #Base SVM

    p=thesispath+"basesvm.png"
    d=DataGenerator()
    neg=d.generate_from_point([-1,-1],40,1,-1)
    pos=d.generate_from_point([1,1],40,1,1)
    labeled=neg+pos
    alg=SVMClassificationAlgorithm(labeled)
    alg.run()
    tmp_plot(alg,labeled,[],p,esvm=False)


    #Base SVR

    p=thesispath+"basesvr.png"
    d=DataGenerator()
    unlabeled=d.generate_from_function(lambda x:-x,50,0.5,-2,2,0)
    import sklearn.svm
    x,y=[u[:1] for u in unlabeled],[u[-1] for u in unlabeled]
    alg=sklearn.svm.SVR(kernel="linear")
    alg.fit(x,y)

    tmp_plot(None,[],unlabeled,p,esvm=False,regrFunc=alg.predict)


    #Base ESVM

    p=thesispath+"baseesvm.png"
    d=DataGenerator()
    neg=d.generate_from_point([-1,-1],40,1,-1)
    pos=d.generate_from_point([1,1],40,1,1)
    labeled=neg+pos
    unlabeled=d.generate_from_function(lambda x:-x,50,0.5,-0.8,0.8,0)

    alg=ESVMClassificationAlgorithm(labeled,unlabeled,1,1,10)
    alg.run()

    tmp_plot(alg,labeled,unlabeled,p,esvm=True)


    #SVM Kernel

    p=thesispath+"svmkernel.png"
    d=DataGenerator()
    neg=d.generate_from_point([-1,0.4],40,1,-1)
    pos=d.generate_from_point([1,1.5],40,0.5,1)
    pos=pos+d.generate_from_point([1,-1.5],40,1.5,1)
    pos2=d.generate_from_point([2.1,0.4],10,1,-1)
    labeled=neg+pos+pos2
    alg=SVMClassificationAlgorithm(labeled,kernel=yaplf.models.kernel.GaussianKernel(7))
    alg.run()

    tmp_plot(alg,labeled,[],p,esvm=False)
"""

    #index_quality

    p=thesispath+"indexquality.png"
    d=DataGenerator()
    neg=d.generate_from_function(lambda x:-1,50,0.5,-2,2,-1)
    pos=d.generate_from_function(lambda x:1,50,0.5,-2,2,1)

    labeled=neg+pos
    unlabeled=d.generate_from_function(lambda x:0,50,0.2,-2,2,0)
    datasetlist=[(labeled,unlabeled)]
    alg1=ESVMClassificationAlgorithm(labeled,unlabeled,1,1,10)
    alg1.run()


    neg=d.generate_from_function(lambda x:-2,50,0.5,-2,2,-1)
    pos=d.generate_from_function(lambda x:2,50,0.5,-2,2,1)

    labeled=neg+pos
    unlabeled=d.generate_from_function(lambda x:3*x,100,1,-2,2,0)


    alg2=ESVMClassificationAlgorithm(labeled,unlabeled,1,1,10)
    alg2.run()
    alglist=[alg1,alg2]
    datasetlist.append((labeled,unlabeled))
    fig,axarr=plt.subplots(1,2,True)
    xr=3
    for i in range(len(alglist)):
        labeled,unlabeled=datasetlist[i]
        dataset=labeled+[LabeledExample(u,0) for u in unlabeled]
        alg=alglist[i]
        plt=axarr[i]
        plt.scatter([d.pattern[0] for d in dataset],[d.pattern[1] for d in dataset],c=[cf_f(d) for d in dataset])
        h=float(2*xr)/len(dataset)
        x=range(-xr,xr)
        y=[-alg.model.decision_function((x_i,0)) for x_i in x]
        xx, yy = np.meshgrid(np.arange(-xr,xr, h),
        np.arange(-xr, xr, h))
        l=[(x[0],x[1]) for x in np.c_[xx.ravel(), yy.ravel()]]
        Z=np.array([alg.model.decision_function(x) for x in l])
        Z = Z.reshape(xx.shape)
        contour_value_eps = [alg.model.tube_radius,-alg.model.tube_radius]
        contour_style = ('--',) * len(contour_value_eps)
        plt.contour(xx, yy, Z,contour_value_eps,linestyles=contour_style,colors="g")
        contour_value_eps = [1,-1]
        contour_style = ('-',) * len(contour_value_eps)
        plt.contour(xx, yy, Z,contour_value_eps,linestyles=contour_style,colors="g")
        plt.contour(xx, yy, Z,[0],linewidths=[2],colors="b")
    fig.savefig(p)
