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

    def generate_dataset(self):
        neg=self.generate([-1,-1],10,0.2)
        pos=self.generate([0.5,0.5],10,0.2)
        labeled=[LabeledExample(i,1)for i in pos]+[LabeledExample(i,-1) for i in neg]
        unlabeled=self.generate([0,0],5,1)
        test_set=[LabeledExample([-1,-1],-1),LabeledExample([0,0],1)]
        return [labeled,unlabeled,test_set]

    def generate(self,center,s,scattering):
        return [[d+r(-scattering,scattering)for d in center] for i in range(s)]

    def test_xor(self):
        labeled,unlabeled,test_set=self.generate_dataset()

        print self.base_classification_test(labeled,unlabeled,test_set)