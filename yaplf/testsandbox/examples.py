from yaplf.data import LabeledExample
from yaplf.algorithms.svm.classification.solvers import *
from yaplf.algorithms.svm.classification import *
from yaplf.models.svm.plot import *
import random
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


    test_set=labeled_dataset[:100]
    training_set=labeled_dataset[301:450]
    unlabeled_dataset=unlabeled_dataset[:1000]
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

#ds=read_webspam()
#write_dataset_temp(ds)
training_set,unlabeled_dataset,test_set=read_dataset_temp()

"""
alg = S3VMClassificationAlgorithm(training_set,unlabeled_dataset,c=1,d=0.25,e=0.25)

alg.run() # doctest:+ELLIPSIS



res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]
print float(sum(res))/(len(test_set)+len(training_set))

alg = SVMClassificationAlgorithm(training_set,c=1)

alg.run() # doctest:+ELLIPSIS
res=[1 for i in test_set if alg.model.compute(i.pattern)==i.label ]

print float(sum(res))/(len(test_set)+len(training_set))"""