from yaplf.data import LabeledExample
import unittest
from yaplf.algorithms.svm.classification import *
import random
r=random.uniform
class Test(unittest.TestCase):
    """Unit tests for S3VMClassifier module of yaplf."""

    def base_classification_test(self,labeled,unlabeled,test_set):
        alg = S3VMClassificationAlgorithm(labeled,unlabeled,c=1,d=0.25,e=0.25)
        alg.run() # doctest:+ELLIPSIS

        correct_guess=[alg.model.compute(i.pattern)==i.label for i in test_set]
        return correct_guess

    def generate_simple_dataset(self):
        """
        Simple dataset centered around three points: one labeled with +1, one with -1
        and a third unlabeled.
        """
        neg=self.generate([-1,-1],10,0.2,-1)
        pos=self.generate([0.5,0.5],10,0.2,1)
        labeled=pos+neg
        unlabeled=self.generate([0,0],5,1)
        test_set=[LabeledExample([-1,-1],-1),LabeledExample([0,0],1)]
        return [labeled,unlabeled,test_set]

    def generate(self,center,size,scattering,label=None):
        """

        :param center: center around which the points are generated
        :param size: number of points of the dataset
        :param scattering: max distance from the center of all the points
        :param label: label of the points
        :return: a list of points of size n

        """

        if label:
            return [LabeledExample([d+r(-scattering,scattering)for d in center],label) for i in range(size)]
        else:
            return [[d+r(-scattering,scattering)for d in center] for i in range(size)]

    def test_simple(self):
        labeled,unlabeled,test_set=self.generate_simple_dataset()

        print self.base_classification_test(labeled,unlabeled,test_set)