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
from os.path import expanduser
import datetime
warnings.simplefilter("error")

home = expanduser("~")

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start
r=randrange_float
class Test(unittest.TestCase):
    """Unit tests for S3VMClassifier module of yaplf."""

    def base_classification(self,labeled,unlabeled,test_set):
        alg = S3VMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=1)
        alg.run() # doctest:+ELLIPSIS
        correct_guess=[alg.model.compute(i.pattern)==i.label for i in test_set]
        return correct_guess







    def generate_simple_dataset(self):
        """
        Simple dataset centered around three points: one labeled with +1, one with -1
        and a third unlabeled.
        """
        neg=self.generate_from_point([0,0],20,2,-1)
        pos=self.generate_from_point([1,1],20,2,1)
        labeled=pos+neg
        #labeled=[LabeledExample([-0.9,-0.9],-1),LabeledExample([0.9,-0.9],1),LabeledExample([-0.9,0.9],1),LabeledExample([0.9,0.9],1)]
        #unlabeled=self.generate_from_point([0.570710,0.570710],300,2,0)
        unlabeled=self.generate_from_function(lambda x:x**3,20,0.1,-7,7)

        #unlabeled=[[-0.1,-0.1],[-0.3,-0.3],[0.1,-0.3],[0.2,-0.4],[0.3,-0.366648]]
        test_set=[LabeledExample([-0.1,-0.1],-1),LabeledExample([0,0],1)]
        return [labeled,unlabeled,test_set]

    def generate_from_point(self,center,size,scattering,label=None):
        """

        :param center: center around which the points are generated
        :param size: number of points of the dataset
        :param scattering: max distance from the center of all the points
        :param label: label of the points
        :return: a list of points of size n

        """

        if label:
            return [LabeledExample([d+r(-scattering,scattering,0.01)for d in center],label) for i in range(size)]
        else:
            return [[d+r(-scattering,scattering,0.01)for d in center] for i in range(size)]

    def generate_from_function(self,fn,size,scattering,int_start,int_end,label=None):
        interval=linspace(int_start,int_end,size)
        p_size=size/len(interval)
        l=[self.generate_from_point([x,fn(x)],p_size,scattering,label) for x in interval]
        return [i for j in l for i in j ]

    """ def test_simple(self):
        d=self.base_classification(*self.generate_simple_dataset())
    """

       # print self.base_classification_test(labeled,unlabeled,test_set)

    def test_tube(self):

      labeled,unlabeled,test_set=self.generate_simple_dataset()
      for i in range(15):
            alg = S3VMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=0.1+i*i, #kernel=yaplf.models.kernel.GaussianKernel(1),
                                              tolerance=0.00001)
            path=str(home)+"/grafici/prova"+str(i)+".jpg"
            import os
            try:
             os.remove(path)
            except OSError:
              pass
            try:
                alg.run() # doctest:+ELLIPSIS
            except Exception as e:
                "Errore"
                continue
            m=alg.model
            intube_post_indices=[i for i in range(len(unlabeled)) if alg.model.intube(unlabeled[i])]
            intube_model_indices=m.in_tube_unlabeled_indices
            try:
                self.assertEqual(intube_model_indices,intube_post_indices)
            except AssertionError as e:
                    file=open(str(home)+"/grafici/dump_error"+str(datetime.datetime.now())+".txt","wb")

                    pickle.dump([labeled,unlabeled,intube_model_indices],file)
                    file.close()
                    raise e

            print alg.model.tube_radius
            self.tmp_plot(alg,labeled,unlabeled,path)



    def tmp_plot(self,alg,labeled,unlabeled,path):


            cf_f=lambda x:("yellow" if x.label==1 else ("blue" if x.label==0 else "red"))
            ll=lambda x,y: alg.model.decision_function((x,y))
            ll=lambda x,y:x+y

            dataset=labeled+[LabeledExample(i,0) for i in unlabeled]
            fig=classification_data_plot(dataset,color_function=cf_f)

            axes = fig.add_subplot(111)
            x=range(-5,5)
            y=[-alg.model.decision_function((x_i,0)) for x_i in x]
            import numpy as np
            h=0.2
            xx, yy = np.meshgrid(np.arange(-5,5, h),
                     np.arange(-5, 5, h))
            l=[(x[0],x[1]) for x in np.c_[xx.ravel(), yy.ravel()]]
            Z=np.array([alg.model.decision_function(x) for x in l])

            Z = Z.reshape(xx.shape)

            contour_value_eps = [alg.model.tube_radius,-alg.model.tube_radius]
            contour_style = ('-',) * len(contour_value_eps)
            plt.contour(xx, yy, Z,contour_value_eps,linestyles=contour_style,colors="r")
            plt.contour(xx, yy, Z,[0],linestyles=contour_style,colors="g")

            fig.savefig(path)

