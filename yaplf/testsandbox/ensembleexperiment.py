__author__ = 'chobeat'

from thesisdraw import *
from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
from ionosphere.ionosphere import *
from yaplf.utility.parameterselection import *
import cmeans
import numpy
import sklearn.ensemble
from itertools import groupby as g
import diabetes

def ensembletreeexp():
    ambiguity_threshold=0.2
    dataset=diabetes.read_diabetes()

    random.shuffle(dataset)
    training_set_ensemble=dataset[:200]
    X=[ example.pattern for example in training_set_ensemble]
    Y=[ example.label for example in training_set_ensemble]

    X_test=[ example.pattern for example in dataset]
    Y_test=[ example.label for example in dataset]

    clf=sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion='gini')
    clf.fit(X,Y)

    dataset_emulated_labels=[(example.pattern,[estimator.predict(example.pattern)[0] for estimator in clf.estimators_]) for example in dataset]
    weights= [float(sum([1 for label in example[1] if label==1]))/len(example[1]) for example in dataset_emulated_labels]
    ambiguous_index=[index for index,w in enumerate(weights) if w<1-ambiguity_threshold and w>ambiguity_threshold]

    def most_common_and_format(L):
        r=max(g(sorted(L)), key=lambda(x, v):(len(list(v)),-L.index(x)))[0]
        if r==0:
            return -1
        else:
            return 1

    labeled=[LabeledExample(dataset_emulated_labels[i][0],most_common_and_format(dataset_emulated_labels[i][1]))
             for i in range(len(dataset_emulated_labels))if i not in ambiguous_index]
    unlabeled=[dataset_emulated_labels[i][0] for i in range(len(dataset_emulated_labels))if i in ambiguous_index]
    y1=[]
    y2=[]
    y3=[]
    fold=5
    x=arange(0.1,0.3,0.02)
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}

    for e_i in x:

            res1=cross_validation(ESVMClassificationAlgorithm, labeled, fold, return_quality_index=True,
                         c=2,e=e_i*len(unlabeled),
                                                                                             d=1,
                                                      kernel=GaussianKernel(1),
                 unlabeled_sample=unlabeled,**kwargs)
            if res1!=0:
                y1.append(res1[0])
                y3.append(res1[2])

    res2=cross_validation(SVMClassificationAlgorithm, labeled, fold, kernel=GaussianKernel(1))
    if res2!=0:
           y2=[res2[0]]*len(x)



    x=map(lambda e_i:e_i*len(unlabeled),x)
    f=figure()
    print len(y1),len(y2),len(y3)
    p=f.add_subplot(111,xlabel='Valore di E')
    p.plot(x,y1,"r-",x,y2,"b-",x,y3,"g-")
    f.savefig("/home/chobeat/git/tesi/esperimenti/diabeensemble.png")

ensembletreeexp()