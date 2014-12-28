from yaplf.data import LabeledExample
from yaplf.algorithms.svm.classification.solvers import *
from yaplf.algorithms.svm.classification import *
from yaplf.models.svm.plot import *
import warnings
from yaplf.graph import *
import random
import matplotlib as plt

import yaplf.models.kernel
warnings.simplefilter("error")
def read_webspam():
    with open("feat.csv","r") as f:
        feat= [i.split(",") for i in f][1:]
        feat_dict={int(i[0]):[float(j) for j in i[2:]] for i in feat}
        keyword_map={"spam":1,"nonspam":-1,"undecided":0}

    with open("labels.txt","r")as f:
        labels=[i.split(" ") for i in f]
        labels_dict={int(i[0]):keyword_map[i[1]] for i in labels}

    labeled_dataset=[LabeledExample(feat_dict[i],j) for i,j in labels_dict.iteritems() if feat_dict.__contains__(i) and j!=0]

    unlabeled_dataset=[j for i,j in feat_dict.iteritems() if not labels_dict.__contains__(i)]


    test_set=labeled_dataset[:150]
    random.shuffle(labeled_dataset)
    training_set=labeled_dataset[151:400]
    unlabeled_dataset=unlabeled_dataset[:150]
    return training_set,unlabeled_dataset,test_set

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


ds=read_webspam()
#write_dataset_temp(ds)
training_set,unlabeled_dataset,test_set=ds #read_dataset_temp()

"""
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
"""
random.shuffle(unlabeled_dataset)
unlabeled_dataset=unlabeled_dataset[70:90]
training_set=training_set[:40]
alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=1,d=1,e=5,kernel=yaplf.models.kernel.GaussianKernel(1))
alg.run()
#print alg.model.intube(unlabeled_dataset[0])

res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
print float(sum(res))/(len(test_set))
"""
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


