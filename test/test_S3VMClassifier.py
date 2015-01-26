import unittest
import random
import warnings
from os.path import expanduser

import matplotlib.pyplot as plt

from yaplf.algorithms.svm.classification import *
from yaplf.graph import classification_data_plot
from yaplf.data import LabeledExample
from yaplf.utility.synthdataset import DataGenerator
import yaplf.models.kernel
warnings.simplefilter("error")
from yaplf.testsandbox.thesisdraw import tmp_plot

class Test(unittest.TestCase):
    """Unit tests for ESVMClassifier module of yaplf."""

    def base_classification(self,labeled,unlabeled,test_set):
        alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=10)
        alg.run() # doctest:+ELLIPSIS
        correct_guess=[alg.model.compute(i.pattern)==i.label for i in test_set]
        return correct_guess

    """ def test_simple(self):
        d=self.base_classification(*self.generate_simple_dataset())
    """

       # print self.base_classification_test(labeled,unlabeled,test_set)

    def test_tube(self):
      d=DataGenerator()
      labeled,unlabeled,test_set=d.generate_simple_dataset()
      alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=15,
                                              #kernel=yaplf.models.kernel.PolynomialKernel(2),

                                              tube_tolerance=0.0000001,debug_mode=True)
      alg.run() # doctest:+ELLIPSIS
      m=alg.model
      intube_post_indices=[i for i in range(len(unlabeled)) if alg.model.intube(unlabeled[i])]
      intube_model_indices=m.in_tube_unlabeled_indices
      self.assertEqual(intube_model_indices,intube_post_indices)





