from thesisdraw import *
from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
from ionosphere.ionosphere import *
from yaplf.utility.parameterselection import *
from cmeans import *

def clustering_experiment(threshold=0.30):
    labeled_dataset=read_iono()
    random.shuffle(labeled_dataset)
    """test_set=labeled_dataset[:50]
    labeled_dataset=labeled_dataset[50:]
    """
    delabeled=[l.pattern for l in labeled_dataset]
    labels=[[1,0] if l.label==1 else [0,1] for l in labeled_dataset]
    fcm=FuzzyCMeans(delabeled,labels,1.2)
    memberships=fcm.membership()
    candidates_indices=[i for i in range(len(labeled_dataset)) if memberships[i][0]>threshold and memberships[i][1]>threshold]
    labeled_without_candidates=[labeled_dataset[j]for j in range(len(labeled_dataset)) if j not in candidates_indices]
    candidates=[labeled_dataset[j].pattern for j in range(len(labeled_dataset)) if j in candidates_indices]
    """
    alg=ESVMClassificationAlgorithm(labeled,candidates,kernel=GaussianKernel(2))
    alg.run()
    print "threshold",threshold,evaluate_classifier(alg.model,test_set),alg.model.quality_index(labeled,candidates)
    """
    def print_res(res,text=""):
        for params,mean_scores,scores in res:
            scores=[s for s in scores if s>0]
            if len(scores)>0:
                print params,text,threshold,mean(scores)
    """
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    res_without_candidates=param_search(SVMClassificationAlgorithm,labeled_without_candidates,{
        "kernel":param_to_kernel(GaussianKernel,[2])}
                 ,c_args=[],c_kwargs=kwargs)
    print_res(res_without_candidates,"senza candidati")"""

    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    import functools
    my_index_quality_evaluation=functools.partial(index_quality_evaluation,labeled_without_candidates,candidates)
    res_with_candidates=param_search(ESVMClassificationAlgorithm,labeled_without_candidates,{"c":[1],
                                                      "e":[0.1*len(candidates),0.2*len(candidates)],
                                                                                             "d":[1],
                                                      "kernel":param_to_kernel(GaussianKernel,[
                                                          3])}
                 ,c_args=[candidates],c_kwargs=kwargs,evaluation_func=my_index_quality_evaluation)

    print_res(res_with_candidates,"con_candidati")


def clustering_2D_draw(labeled_dataset,threshold):
    random.shuffle(labeled_dataset)
    """test_set=labeled_dataset[:50]
    labeled_dataset=labeled_dataset[50:]
    """
    delabeled=[l.pattern for l in labeled_dataset]
    labels=[[1,0] if l.label==1 else [0,1] for l in labeled_dataset]
    fcm=FuzzyCMeans(delabeled,labels,1.2)
    memberships=fcm.membership()
    candidates_indices=[i for i in range(len(labeled_dataset)) if memberships[i][0]>threshold and memberships[i][1]>threshold]
    labeled_without_candidates=[labeled_dataset[j]for j in range(len(labeled_dataset)) if j not in candidates_indices]
    candidates=[labeled_dataset[j].pattern for j in range(len(labeled_dataset)) if j in candidates_indices]
    print len(labeled_without_candidates),len(candidates)
    plot_data(labeled_without_candidates,candidates,"plot_candidates"+str(threshold)+".jpg")

d=DataGenerator()
labeled_dataset,unlabeled=d.generate_normal_dataset(2)

for i in [0.01,0.05,0.2,0.3,0.4,0.45]:
    clustering_2D_draw(labeled_dataset,i)

