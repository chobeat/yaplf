import unittest
import random
import warnings
from os.path import expanduser

import matplotlib.pyplot as plt

from yaplf.algorithms.svm.classification import *
from yaplf.graph import classification_data_plot
from yaplf.data import LabeledExample
from yaplf.utility.synthdataset import DataGenerator
warnings.simplefilter("error")
from yaplf.testsandbox.thesisdraw import tmp_plot
home = expanduser("~")

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start
r=randrange_float
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
      for i in range(10):
            alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=10+i*2,
                                              #kernel=yaplf.models.kernel.GaussianKernel(1),
                                              regrKernel="rbf",
                                              tube_tolerance=0.0000001,debug_mode=True)
            path=str(home)+"/grafici/prova"+str(i)+".jpg"

            import os
            try:
             os.remove(path)
            except OSError:
                pass
            alg.run()
            try:
                alg.run() # doctest:+ELLIPSIS
            except Exception as e:
                print e
                continue
            m=alg.model
            """
            intube_post_indices=[i for i in range(len(unlabeled)) if alg.model.intube(unlabeled[i])]
            intube_model_indices=m.in_tube_unlabeled_indices
            try:
                self.assertEqual(intube_model_indices,intube_post_indices)
            except AssertionError as e:
                    file=open(str(home)+"/grafici/dump_error"+str(datetime.datetime.now())+".txt","wb")

                    pickle.dump([labeled,unlabeled,intube_model_indices,alg.model],file)
                    file.close()

                    self.tmp_plot(alg,labeled,unlabeled,str(home)+"/grafici/errore.jpg")
            """
            f=lambda x:alg.model.regrPredict(x)
            print "calcolo ",alg.model.regrPredict([[1,1]])
            print "calcolo ",alg.model.regrPredict([[-2,2]])

            print "calcolo ",alg.model.regrPredict([[2,2]])

            tmp_plot(alg,labeled,unlabeled,path,f)




