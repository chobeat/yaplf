__author__ = 'chobeat'

from yaplf.data import LabeledExample
from numpy import arange,linspace
from yaplf.data import LabeledExample
import random
import numpy as np
def randrange_float(start, stop, step):
   return random.randint(0, int((stop - start) / step)) * step + start
r=randrange_float
class DataGenerator():


    def generate_normal_dataset(self,size=50,d=20):
        neg_center= np.random.normal(-0.5, 0.8, d)
        pos_center=np.random.normal(0.5, 0.8, d)

        neg=self.generate_from_point(neg_center,size,0.8,-1)
        pos=self.generate_from_point(pos_center,size,0.8,1)
        labeled=pos+neg

        unlabeled_center=np.random.normal(0, 0.5, d)
        unlabeled=self.generate_from_point(unlabeled_center,size,0.2,0)
        return [labeled,unlabeled]

    def generate_asymmetric_dataset(self,d=40):
        neg_center= np.random.normal(-3, 0.2, d)
        pos_center=np.random.normal(3, 0.2, d)

        neg=self.generate_from_point(neg_center,250,2,-1)
        pos=self.generate_from_point(pos_center,50,2,1)
        labeled=pos+neg

        unlabeled_center=np.random.normal(0, 0.5, d)
        unlabeled=self.generate_from_point(unlabeled_center,20,0.2,0)

        return [labeled,unlabeled]

    def generate_cleaning_dataset(self,d=2):
        neg_center= np.random.normal(-1, 0.2, d)
        pos_center=np.random.normal(1, 0.2, d)
        neg_amb_center=np.random.normal(-0.2,0.2,d)
        pos_amb_center=np.random.normal(0.2,0.2,d)
        neg=self.generate_from_point(neg_center,20,0.3,-1)+self.generate_from_point(neg_amb_center,5,0.5,-1)
        pos=self.generate_from_point(pos_center,20,0.3,1)+self.generate_from_point(pos_amb_center,5,0.5,1)
        labeled=pos+neg

        unlabeled_center=np.random.normal(0, 0.1, d)
        unlabeled=self.generate_from_point(unlabeled_center,20,0.5,0)

        return [labeled,unlabeled]


    def generate_simple_dataset(self,size):
        """
        Simple dataset centered around three points: one labeled with +1, one with -1
        and a third unlabeled.
        """
        neg=self.generate_from_point([0.4,0.8],size,0.3,-1)
        pos=self.generate_from_point([-0.3,-0.3],size,0.6,1)
        #neg=self.generate_from_function(lambda x:x+1,20,0.5,-2,2,-1)

        #pos=self.generate_from_function(lambda x:0.9*x-1,200,0.5,-2,2,1)


        labeled=pos+neg
        #labeled=[LabeledExample([-0.9,-0.9],-1),LabeledExample([0.9,-0.9],1),LabeledExample([-0.9,0.9],1),LabeledExample([0.9,0.9],1)]
        unlabeled=self.generate_from_point([0,0.4],size,0.3,0)
        #unlabeled=self.generate_from_function(lambda x:0.8*x-0.5,100,0.3,-2,2)
        #unlabeled=[[-0.1,-0.1],[-0.3,-0.3],[0.1,-0.3],[0.2,-0.4],[0.3,-0.366648]]

        return [labeled,unlabeled]

    def generate_perf_dataset(self):
        neg=self.generate_from_function(lambda x:+0.8,20,0.2,-1,1,-1)
        pos=self.generate_from_function(lambda x:-0.8,20,0.2,-1,1,1)

        pos=pos+self.generate_from_function(lambda x:0.5*x-0.6,10,0.5,-1,1,1)
        labeled=pos+neg
        central=self.generate_from_function(lambda x:0,50,0.1,-1,1)+self.generate_from_function(lambda x:0,20,0.6,-1,1)
        return labeled,central

    def generate_leap_dataset(self):
        neg=self.generate_from_function(lambda x:+0.8,20,0.2,-1,1,-1)
        pos=self.generate_from_function(lambda x:-0.8,20,0.2,-1,1,1)
        labeled=pos+neg
        central=self.generate_from_function(lambda x:0,50,0.1,-1,1)
        upper=self.generate_from_function(lambda x:0.8,5,0.2,-1,1)
        lower=self.generate_from_function(lambda x:-0.8,5,0.2,-1,1)
        unlabeled=central+upper+lower
        return labeled,unlabeled

    def generate_ensemble_dataset(self):
        neg=self.generate_from_function(lambda x:+0.8,100,0.5,-1,1,-1)
        pos=self.generate_from_function(lambda x:-0.8,100,0.5,-1,1,1)
        labeled=pos+neg
        unlabeled=self.generate_from_function(lambda x:0,50,0.3,-1,1)
        return labeled,unlabeled


    def generate_weighted_dataset(self):
        neg=self.generate_from_function(lambda x:+0.8,20,0.2,-1,1,-1)
        pos=self.generate_from_function(lambda x:-0.8,20,0.2,-1,1,1)
        labeled=pos+neg

        unlabeled=self.generate_from_function(lambda x:0,50,0.2,-1,1)
        def w_l(y):
            if y>0:
                return (1-y,y)
            else:
                return (-y,1+y)


        w=[w_l(s[1]) for s in unlabeled]
        l=[x[0] for x in w ]
        r=[x[1] for x in w ]


        return [labeled,unlabeled,l,r]


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
