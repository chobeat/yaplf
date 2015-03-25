from thesisdraw import *
from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
from ionosphere.ionosphere import *
from yaplf.utility.parameterselection import *
import cmeans
import numpy


def clustering_2D_draw(labeled_dataset,threshold):
    random.shuffle(labeled_dataset)
    """test_set=labeled_dataset[:50]
    labeled_dataset=labeled_dataset[50:]
    """
    delabeled=[l.pattern for l in labeled_dataset]
    labels=[[1,0] if l.label==1 else [0,1] for l in labeled_dataset]
    fcm=cmeans.FuzzyCMeans(delabeled,labels,1.2)
    memberships=fcm.membership()
    [is_anom_candidate(fcm.centers(),memberships[i],labeled_dataset[i].pattern,threshold)  for i in range(len(labeled_dataset))]
    candidates_indices=[i for i in range(len(labeled_dataset)) if memberships[i][0]>threshold and memberships[i][1]>threshold]

    labeled_without_candidates=[labeled_dataset[j]for j in range(len(labeled_dataset)) if j not in candidates_indices]
    candidates=[labeled_dataset[j].pattern for j in range(len(labeled_dataset)) if j in candidates_indices]
    print len(labeled_without_candidates),len(candidates)
    plot_data(labeled_without_candidates,candidates,"plot_candidates"+str(threshold)+".jpg")


def clustering_experiment(labeled_dataset,fold,test_set,e_range,threshold=0.30):
    """test_set=labeled_dataset[:50]
    labeled_dataset=labeled_dataset[50:]
    """
    delabeled=[l.pattern for l in labeled_dataset]
    labels=[[1,0] if l.label==1 else [0,1] for l in labeled_dataset]
    fcm=cmeans.FuzzyCMeans(delabeled,labels,1.2)
    memberships=fcm.membership()
    candidates_indices=[i for i in range(len(labeled_dataset)) if memberships[i][0]>threshold and memberships[i][1]>threshold]
    labeled_without_candidates=[labeled_dataset[j]for j in range(len(labeled_dataset)) if j not in candidates_indices]
    candidates=[labeled_dataset[j].pattern for j in range(len(labeled_dataset)) if j in candidates_indices]
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    import functools
    y1=[]
    y2=[]
    for e_i in e_range:
        res_with_candidates=cross_validation(ESVMClassificationAlgorithm, labeled_without_candidates, fold, test_set,
                                             return_quality_index=True,
                         c=4,e=e_i*len(candidates),
                                                                                             d=1,
                                                      kernel=GaussianKernel(2),
                 unlabeled_sample=candidates,**kwargs)
        print res_with_candidates
        y1.append(res_with_candidates[0])
        y2.append(res_with_candidates[2])

    res_svm_without=cross_validation(SVMClassificationAlgorithm,labeled_without_candidates,fold, test_set,kernel=GaussianKernel(2),
                 **kwargs)
    res_svm_with=cross_validation(SVMClassificationAlgorithm,labeled_dataset,fold, test_set,kernel=GaussianKernel(2),
                 **kwargs)


    return y1,y2,res_svm_with,res_svm_without,len(candidates)

def is_anom_candidate(centers,membership,coords,threshold):
    def dist(x,y):
        return numpy.sqrt(numpy.sum((x-y)**2))
    distances_from_centers=[dist(numpy.array(c),numpy.array(coords)) for c in centers]
    max_memb_index=[i for i in range(len(membership)) if membership[i]==max(membership)]
    min_dist_index=[j for j in range(len(distances_from_centers)) if distances_from_centers[j]==min(distances_from_centers)]
    if max_memb_index!=min_dist_index:
        print "anomalo",coords

"""
d=DataGenerator()
labeled_dataset=d.generate_from_point((-1,1),50,1,1)+d.generate_from_point((0.1,2),20,1,-1)
"""
def svm_comparison_experiment(fold=5):
    labeled_dataset=read_iono()
    random.shuffle(labeled_dataset)
    test_set=labeled_dataset[:100]
    labeled_dataset=labeled_dataset[100:]
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    e_x=arange(0.01,0.35,0.02)
    results=clustering_experiment(labeled_dataset,fold,test_set,e_x,threshold=0.3)
    print e_x
    import matplotlib.figure
    f=figure()
    p=f.add_subplot(111,xlabel='Valore di E')
    print results[2][0]*len(e_x)

    e_x=[i*results[4] for i in e_x]
    p.plot(e_x,results[0],"y-",e_x,results[1],"g-",e_x,[results[2][0]]*len(e_x),"r--",e_x,[results[3][0]]*len(e_x),"b--")
    f.savefig("/home/chobeat/git/tesi/esperimenti/clusteringvariaresogliatestsetfisso.png")
svm_comparison_experiment(7)

