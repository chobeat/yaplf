__author__ = 'chobeat'

from yaplf.data import LabeledExample
from numpy import arange,linspace
from yaplf.data import LabeledExample
import random
def randrange_float(start, stop, step):
   return random.randint(0, int((stop - start) / step)) * step + start
r=randrange_float
class DataGenerator():


    def generate_simple_dataset(self):
        """
        Simple dataset centered around three points: one labeled with +1, one with -1
        and a third unlabeled.
        """
        #neg=self.generate_from_point([-1,-1],10,1.5,-1)
        #pos=self.generate_from_point([3,3],10,1.5,1)
        neg=self.generate_from_function(lambda x:x+1,20,0.2,-2,2,-1)

        pos=self.generate_from_function(lambda x:x-1,20,0.2,-2,2,1)


        labeled=pos+neg
        #labeled=[LabeledExample([-0.9,-0.9],-1),LabeledExample([0.9,-0.9],1),LabeledExample([-0.9,0.9],1),LabeledExample([0.9,0.9],1)]
        #unlabeled=self.generate_from_point([0.570710,0.570710],20,2,0)
        unlabeled=self.generate_from_function(lambda x:x-0.5,30,0.1,-2,2)
        #unlabeled=[[-0.1,-0.1],[-0.3,-0.3],[0.1,-0.3],[0.2,-0.4],[0.3,-0.366648]]
        test_set=[LabeledExample([-0.1,-0.1],-1),LabeledExample([0,0],1)]
        return [labeled,unlabeled,test_set]


    def generate_weighted_dataset(self):
        neg=self.generate_from_function(lambda x:+1,20,0.2,-2,2,-1)

        pos=self.generate_from_function(lambda x:-1,20,0.2,-2,2,1)


        labeled=pos+neg
        unlabeled=self.generate_from_function(lambda x:0,30,0.3,-2,2)
        def w_l(x):
            if x<0:
                return 1+x
            else:
                return -x
        def w_r(x):
            if x>=0:
                return 1-x
            else:
                return x

        l=[w_l(s[1]) for s in unlabeled]
        r=[w_r(s[1]) for s in unlabeled]
        print l,r
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
