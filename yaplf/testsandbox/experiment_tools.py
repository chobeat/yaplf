import numpy as np
from yaplf.data import LabeledExample
from sklearn import metrics
from os.path import expanduser
from yaplf.algorithms.svm.classification import *
from yaplf.testsandbox.thesisdraw import tmp_plot
import math
import os
from ensemble import *
import csv
import os
from subprocess import call
def cross_validation_split(labeled,fold):

    size=len(labeled)/fold
    return [(labeled[i*size:(i+1)*size],labeled[:i*size]+labeled[(i+1)*size:]) for i in range(fold)]

def cross_validation(cls,dataset,fold,three_way=False,*args,**kwargs):
    grouped_dataset=cross_validation_split(dataset,fold)
    precision_sum=0
    ambiguous=float(0)
    errors=float(0)
    for test_set,training_set in grouped_dataset:
        alg=cls(training_set,*args,**kwargs)

        try:
            alg.run()
        except Exception,err:
            print err
            errors+=1
            continue
        curr_precision,curr_ambiguous=test_error(alg.model,test_set,three_way)
        if curr_precision>0:
            precision_sum=precision_sum+curr_precision
            ambiguous+=curr_ambiguous
        else:
            errors+=1
    if fold-errors>0:
        average_precision=precision_sum/(fold-errors)
        average_ambiguous=ambiguous/(fold-errors)
        return  average_precision,average_ambiguous
    else:
        return "errore che boh"


def split_by_label(labeled):
    pos,neg=[],[]
    for i in labeled:
        (pos if i.label==1 else neg).append(i)

    return pos,neg

def persist_result(res,name,dataset_type,annotation=None,overwrite=True):

    path="/home/chobeat/git/tesi/esperimenti/"+name

    if overwrite and os.path.isfile(path+".txt"):
        os.remove(path)
    elif os.path.isfile(path+".txt"):
         i=2
         t_path=path+"_"+str(i)
         while(os.path.isfile(t_path+".txt")):
             i+=1
             t_path=path+"_"+str(i)
         path=t_path
    path=path+".txt"
    with open(path,"wb") as f:
        f.write("dataset: "+dataset_type+"\n")
        f.write(annotation+"\n")

        writer=csv.writer(f,delimiter=" ")
        writer.writerow(res)
def read_webspam():
    path="/home/chobeat/git/yaplf/yaplf/testsandbox/"
    keyword_map={"spam":1,"nonspam":-1,"undecided":0}
    with open(path+"feat.csv","r") as f:

        feat= [i.split(",") for i in f][1:]
        feat_dict={int(i[0]):[float(j) for j in i[2:]] for i in feat}

    with open(path+"decided_labels.txt","r")as f:
        labels=[i.split(" ") for i in f]
        labels_dict={int(i[0]):keyword_map[i[1]] for i in labels}

    with open(path+"undecided_labels.txt","r") as f:

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

def logistic_scaling_function(x,model,beta):
    def sigmoid(x,beta):
        return 1 / (1 + math.exp(-x*beta))

    dist=model.decision_function(x)
    a=sigmoid(dist,beta)

    return a


def default_scaling_function(x,model,*args):
    return model.compute(x)


def index_quality_evaluation(labeled,unlabeled,model,*args,**kwargs):
    return model.quality_index(labeled,unlabeled)

def evaluate_classifier(model,test_set,scaling_function=default_scaling_function,scaling_params=[]):

    def format_data(label_list):
        return [(l+abs(l))/2 for l in label_list]

    y_true=format_data([x.label for x in test_set])
    y_eval=[scaling_function(x.pattern,model,*scaling_params) for x in test_set]
    y_eval=format_data(y_eval)

    diff=[(y_eval[i],y_true[i]) for i in range(len(y_eval)) if y_eval[i]!=y_true[i]]

    tmp_path="./temp.csv"
    with open(tmp_path,"wb") as f:
        example_writer=csv.writer(f, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(y_eval)):
            example_writer.writerow(("example_"+str(i),y_true[i],y_eval[i]))

    res=os.popen(" cat ./temp.csv \
  | sed 's/NONSPAM/0/g' | sed 's/SPAM/1/g' \
  | grep -v '^#' | awk '{print $2,$3}' | /home/chobeat/git/yaplf/yaplf/testsandbox/perf -PRF -AUC -plot pr").read()

    roc=float(res.split("\n")[-2].replace("ROC    ",""))

    os.remove(tmp_path)

    return roc

def test_error(model,test_set,three_way=False):
    ambiguous_count=0
    correct_count=0

    for x in test_set:
        pattern,true_label=x.pattern,x.label
        if three_way:
            label=model.three_way_classification(pattern)
        else:
            label=model.compute(pattern)
        if label==0:
            ambiguous_count+=1
        else:
            if label==true_label:
                correct_count+=1
    if len(test_set)-ambiguous_count>0:
        test_error=float(correct_count)/(len(test_set)-ambiguous_count)
        return test_error,ambiguous_count
    else:
        return 0,0

def start_experiment(alg,labeled,unlabeled,test_set=None,path=None,ESVM=True):

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

         tmp_plot(alg,labeled,unlabeled,path,esvm=ESVM)
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


def read_webspam_with_votelist():
    labeled,undecided=read_webspam()
    unlabeled=[i[0] for i in undecided]
    voteList=label_to_weight([i[1] for i in undecided])
    ambiguous=[collections.Counter(i[1])["U"]<float(len(i[1]))/2 for i in undecided]

    return labeled,unlabeled,voteList[0],ambiguous
