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



def tmp_plot(alg,labeled,unlabeled,path=None,esvm=True,regrFunc=None):

            dataset=labeled+[LabeledExample(i,0) for i in unlabeled]
            fig=classification_data_plot(dataset,color_function=cf_f)
            xr=1
            axes = fig.add_subplot(111)
            x=range(-xr,xr)
            y=[-alg.model.decision_function((x_i,0)) for x_i in x]

            h=0.2
            xx, yy = np.meshgrid(np.arange(-xr,xr, h),
                     np.arange(-xr, xr, h))
            l=[(x[0],x[1]) for x in np.c_[xx.ravel(), yy.ravel()]]

            Z=np.array([alg.model.decision_function(x) for x in l])

            Z = Z.reshape(xx.shape)

            if esvm:
                 contour_value_eps = [alg.model.tube_radius,-alg.model.tube_radius]

                 contour_style = ('--',) * len(contour_value_eps)
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
                plt.plot(xx,yy,linestyle="--",linewidth=2,color="orange")
            if path:
                fig.savefig(path,bbox_inches='tight')
            else:
                fig.plot()
            plt.close()
if __name__=="__main__":

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
