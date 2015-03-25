__author__ = 'chobeat'

from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from yaplf.testsandbox.experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
import csv
from yaplf.utility.parameterselection import *
from yaplf.testsandbox.cmeans import *
import numpy

def read_diabetes():

    path="/home/chobeat/git/yaplf/yaplf/testsandbox/"

    with open(path+"diabetes_scale.txt","r") as f:
        reader=csv.reader(f)
        res=[row[0].split(" ") for row in reader]
        labels=[int(r[0]) for r in res]
        def fill(x):
            if x ==" " or x=="":
                return 0.5
            else:
                return float(x)

        patterns=[[fill(p[2:]) for p in r[1:9]] for r in res]

        res=[LabeledExample(p,l) for p,l in zip(patterns,labels)]

    return res
if __name__=="__main__":


    x=[(i.pattern) for i in read_diabetes()]
    print type(x[0][0])
