__author__ = 'chobeat'

from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from yaplf.testsandbox.experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
import csv
from yaplf.utility.parameterselection import *
from yaplf.testsandbox.cmeans import *
import numpy



def read_iono():
    path="/home/chobeat/git/yaplf/yaplf/testsandbox/"
    keyword_map={"g":1,"b":-1}
    with open(path+"ionosphere/ionosphere.data","r") as f:
        reader=csv.reader(f)
        reader=list(reader)
        res=[LabeledExample([float(v)for v in row[:-1]],keyword_map[row[-1]])for row in reader]


    return res
"con soglia a 0.125, performance massima 0.96491"
def iono_cluster_param_search(threshold=0.125):
    labeled_dataset=read_iono()

    search_kernel=yaplf.models.kernel.GaussianKernel
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    delabeled=[l.pattern for l in labeled_dataset]
    labels=[[1,0] if l.label==1 else [0,1] for l in labeled_dataset]
    fcm=FuzzyCMeans(delabeled,labels,1.2)
    memberships=fcm.membership()
    candidates_indices=[i for i in range(len(labeled_dataset)) if memberships[i][0]>threshold and memberships[i][1]>threshold]
    labeled_without_candidates=[labeled_dataset[j]for j in range(len(labeled_dataset)) if j not in candidates_indices]
    candidates=[labeled_dataset[j].pattern for j in range(len(labeled_dataset)) if j in candidates_indices]
    import functools
    print len(labeled_dataset),len(labeled_without_candidates),len(candidates)
    res0=param_search(ESVMClassificationAlgorithm,labeled_without_candidates,{"c":[2],
                                                    "e":[0.10*len(candidates),0.15*len(candidates),0.05*len(candidates)],"d":[1,2,3],
                                                    "kernel":param_to_kernel(search_kernel,[1,2,3])}
                 ,c_args=[candidates],c_kwargs=kwargs)

    res_svm=param_search(SVMClassificationAlgorithm,labeled_without_candidates,{"c":[1],
                                                    "kernel":param_to_kernel(search_kernel,[2])},[],{}
                 )


    for r in res0+res_svm:
        pos_scores=[x for x in r[2] if x>0]
        if len(pos_scores)>0:
            print r[0],sum(pos_scores)/len(pos_scores )




iono_cluster_param_search()