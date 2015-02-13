import numpy as np
from yaplf.data import LabeledExample
from sklearn import metrics
from os.path import expanduser
from yaplf.algorithms.svm.classification import *
from yaplf.testsandbox.thesisdraw import tmp_plot
import os
from ensemble import *


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

def persist_result(res,name,annotation=None):
    path="/home/chobeat/git/tesi/esperimenti"+name+".txt"
    with open(path,"wb") as f:
        f.write(annotation)
        f.write(res)
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

def evaluate_classifier(model,test_set):
    def format_data(label_list):
        return [(l+abs(l))/2 for l in label_list]

    y_true=format_data([x.label for x in test_set])
    y_eval=[model.compute(x.pattern) for x in test_set]
    y_eval=format_data(y_eval)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(y_true), np.array(y_eval), pos_label=1)

    auc=metrics.auc(fpr, tpr)

    return auc


def start_experiment(alg,labeled,unlabeled,test_set=None,path=None):

    try:
            alg.run() # doctest:+ELLIPSIS
    except Exception as e:
          print e
          return
    m=alg.model
    if path:
         try:
            os.remove(path)
         except OSError:
                pass

         tmp_plot(alg,labeled,unlabeled,path)
    if test_set:
        return evaluate_classifier(m,test_set)

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

