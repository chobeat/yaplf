from yaplf.data import LabeledExample
from os.path import expanduser
from yaplf.algorithms.svm.classification.solvers import *
from yaplf.algorithms.svm.classification import *
from yaplf.models.svm.plot import *
import warnings
from yaplf.graph import *
import random
import matplotlib as plt
import yaplf.models.kernel
from yaplf.utility.synthdataset import DataGenerator
from yaplf.testsandbox.thesisdraw import tmp_plot
import os
import collections
warnings.simplefilter("error")

def read_webspam():

    keyword_map={"spam":1,"nonspam":-1,"undecided":0}
    with open("feat.csv","r") as f:

        feat= [i.split(",") for i in f][1:]
        feat_dict={int(i[0]):[float(j) for j in i[2:]] for i in feat}

    with open("decided_labels.txt","r")as f:
        labels=[i.split(" ") for i in f]
        labels_dict={int(i[0]):keyword_map[i[1]] for i in labels}

    with open("undecided_labels.txt","r") as f:

        undecided_labels=[i.split(" ") for i in f]
        def extract(x):
            return [vote.split(":")[1] for vote in x.strip().split(",")]

        undecided={int(l[0]):collections.Counter(extract(l[3]))for l in undecided_labels }

    labeled_dataset=[LabeledExample(feat_dict[i],j) for i,j in labels_dict.iteritems()
                     if i in feat_dict and j!=0]


    unlabeled_dataset=[(feat_dict[int(i)],j) for i,j in undecided.iteritems() if i in feat_dict]

    return labeled_dataset,unlabeled_dataset

import pickle
def write_dataset_temp(dataset):
    file=open("temp.txt","wb")
    pickle.dump(dataset,file)
    file.close()

def read_dataset_temp():
    file=open("temp.txt","r")
    r=pickle.load(file)
    file.close()

    return r

home = expanduser("~")

def main_example():

      d=DataGenerator()
      labeled,unlabeled,l,r=d.generate_weighted_dataset()
      for i in range(10):
            alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=10+i*5,l_weight=l,r_weight=r,
                                              #kernel=yaplf.models.kernel.PolynomialKernel(2),

                                              tube_tolerance=0.0000001,debug_mode=True)
            path=str(home)+"/grafici/prova"+str(i)+"w.jpg"
            start_experiment(alg,path)

def start_experiment(alg,path,labeled,unlabeled):
    try:
            os.remove(path)
    except OSError:
                pass
    try:
            alg.run() # doctest:+ELLIPSIS
    except Exception as e:
          print e
          return
    m=alg.model
    tmp_plot(alg,labeled,unlabeled,path)

def label_to_weight(votes_list):
    weight_map={"S":1,"N":0,"U":0.5,"B":0.8}
    r=[]
    for example in votes_list:
        x=[]
        for label,count in example.iteritems():
            x=x+([weight_map[label]]*count)
        r.append(mean(x))

    l=[1-w for w in r]
    return (l,r)

def weight_experiment():
     d=DataGenerator()
     pos=d.generate_from_point([0,2],1,0,1)
     neg=d.generate_from_point([0,-1],1,0,-1)
     labeled=pos+neg
     unlabeled=[[0,0.5],[0,0.6],[0,0.65],[0,0.7]]
     alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=1,d=1,e=0.3,
                                              #kernel=yaplf.models.kernel.PolynomialKernel(2),

                                              tube_tolerance=0.0000001,debug_mode=True)
     path=str(home)+"/grafici/weight1.jpg"
     start_experiment(alg,path,labeled,unlabeled)
     print "secondo"

     path=str(home)+"/grafici/weight2.jpg"
     alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=10,d=1,e=0.3,l_weight=[1]*len(unlabeled),r_weight=[0.5]*len(unlabeled),
                                              #kernel=yaplf.models.kernel.PolynomialKernel(2),

                                              tube_tolerance=0.0000001,debug_mode=True)
     start_experiment(alg,path,labeled,unlabeled)
     print "terzo"

     path=str(home)+"/grafici/weight3.jpg"
     alg = ESVMClassificationAlgorithm(labeled,unlabeled,c=10,d=1,e=0.3,l_weight=[1]*len(unlabeled),r_weight=[0.001]*len(unlabeled),
                                              #kernel=yaplf.models.kernel.PolynomialKernel(2),

                                              tube_tolerance=0.0000001,debug_mode=True)
     start_experiment(alg,path,labeled,unlabeled)

def cross_validation_split(labeled,fold):

    size=len(labeled)/fold
    return [(labeled[i*size:(i+1)*size],labeled[:i*size]+labeled[(i+1)*size:]) for i in range(fold)]

def cross_validation(cls,dataset,fold,*args,**kwargs):
    grouped_dataset=cross_validation_split(dataset,fold)
    precision_sum=0
    for test_set,training_set in grouped_dataset:
        alg=cls(training_set,*args,**kwargs)
        try:
            alg.run()
        except Exception,err:
            print err
            continue
        curr_precision=float(sum([1 for i in test_set if alg.model.compute(i.pattern)==i.label ]))/len(test_set)
        precision_sum=precision_sum+curr_precision

    average_precision=precision_sum/fold
    return  average_precision


def webspam_experiment1():

    decided,undecided=read_dataset_temp()
    unlabeled=[i[0] for i in undecided]
    decided=decided[:500]
    random.shuffle(decided)

    """print "SVM Base:",cross_validation(SVMClassificationAlgorithm, yaplf.models.kernel.GaussianKernel(2),decided,10)"""
    print "ESVM",cross_validation(ESVMClassificationAlgorithm,yaplf.models.kernel.GaussianKernel(2),decided,10,unlabeled)

    """   to_print=[]
    for c_i in [1]:
        for g_i in [1]:
            for e_i in [5]:

                alg = ESVMClassificationAlgorithm(training_set,unlabeled,c=c_i,d=1,e=e_i,
                                              kernel=yaplf.models.kernel.GaussianKernel(g_i),

                                              tube_tolerance=0.0000001,debug_mode=True)
                try:
                    alg.run() # doctest:+ELLIPSIS
                except Exception,err:
                    print err
                    print "Errore con c={0}, d={1}, e={2}".format(c_i,1,e_i)
                    continue
                res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]

                to_print.append(["c: "+str(c_i),"d: "+str(1),"e: "+str(e_i),float(sum(res))/(len(test_set))])
                print "finito"
    for i in to_print:
        print i
    """


def webspam_weight_experiment1():

    decided,undecided=read_dataset_temp()
    unlabeled=[i[0] for i in undecided]
    l_weight,r_weight=label_to_weight([u[1] for u in undecided])
    random.shuffle(decided)

    decided=decided[:200]

    print "SVM Base:",cross_validation(SVMClassificationAlgorithm, decided,10,kernel=yaplf.models.kernel.GaussianKernel(2))
    print "ESVM",cross_validation(ESVMClassificationAlgorithm,decided,10,unlabeled,c=1,d=1,e=30,kernel=yaplf.models.kernel.GaussianKernel(2),
                                    l_weight=l_weight,r_weight=r_weight)


def webspam_weight_experiment2():

    polarization_threshold=0.5
    decided,undecided=read_dataset_temp()
    unlabeled=[i[0] for i in undecided]

    l_weight,r_weight=label_to_weight([u[1] for u in undecided])
    threshold_undecided_set=[LabeledExample(u[0][0],(1 if u[1]>=polarization_threshold else -1)) for u in zip(undecided,r_weight)]

    random.shuffle(decided)

    decided=decided[:500]
    for e_i in [0.05,0.1,0.2,0.3]:
        alg=ESVMClassificationAlgorithm(decided,unlabeled,c=1,d=1,e=e_i*len(unlabeled),kernel=yaplf.models.kernel.GaussianKernel(2),
                                        l_weight=l_weight,r_weight=r_weight)
        alg.run()
        print e_i,alg.model.in_tube_ratio()

webspam_weight_experiment2()

"""
ds=read_webspam()
#write_dataset_temp(ds)
training_set,unlabeled_dataset,test_set=ds #read_dataset_temp()

to_print=[]
for c_i in [1]:
    for d_i in [1]:
        for e_f in [1.02]:
            e_i=e_f*len(unlabeled_dataset)
            alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=c_i,d=d_i,e=e_i)
            alg.run() # doctest:+ELLIPSIS
            res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
            to_print.append(["c: "+str(c_i),"d: "+str(d_i),"e: "+str(e_i),float(sum(res))/(len(test_set))])

for i in to_print:
    print i

random.shuffle(unlabeled_dataset)
unlabeled_dataset=unlabeled_dataset[70:90]
training_set=training_set[:40]
alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=1,d=1,e=5,kernel=yaplf.models.kernel.GaussianKernel(1))
alg.run()
#print alg.model.intube(unlabeled_dataset[0])

res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
print float(sum(res))/(len(test_set))
alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=f*1.5,d=1,e=len(unlabeled_dataset))
alg.run()
res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
print float(sum(res))/(len(test_set))


alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=f*0.5,d=1,e=len(unlabeled_dataset))
alg.run()
res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
print float(sum(res))/(len(test_set))


alg = SVMClassificationAlgorithm(training_set,c=1)

alg.run() # doctest:+ELLIPSIS
res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
"""


