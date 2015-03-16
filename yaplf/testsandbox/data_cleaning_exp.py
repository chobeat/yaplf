from functools import partial
from thesisdraw import *
from yaplf.utility.synthdataset import *
from multiprocessing.pool import ThreadPool
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function,split_by_label
def data_cleaning_experiment(dataset,cycles=3,e=0.1):
    labeled,unlabeled,a,b=dataset

    random.shuffle(labeled)
    pos,neg=split_by_label(labeled)
    labeled=pos[:200]+neg[:300]
    random.shuffle(labeled)
    test_set=labeled[400:500]
    labeled=labeled[:400]
    unlabeled=unlabeled[:200]
    neg_test_perc=float(len(split_by_label(test_set)[1]))/len(test_set)
    res=[]
    curr_cycle=0
    while(curr_cycle<cycles):
        print len(labeled),len(unlabeled)
        alg=ESVMClassificationAlgorithm(labeled,unlabeled,e=e*len(unlabeled),kernel=GaussianKernel(2),tube_tolerance=0.01)
        try:
            alg.run()
        except Exception as exc:
            print exc
            e=e*0.9
            continue

        res.append(("index quality",alg.model.quality_index(labeled,unlabeled),
                    "auc",evaluate_classifier(alg.model,test_set),
                    "precision",    float(len([1 for t in test_set if alg.model.compute(t.pattern)==t.label]))/len(test_set),
                    "negatives perc",neg_test_perc
                    ))
        #tmp_plot(alg,labeled,unlabeled,"tmpdataplot"+str(c)+".jpg")

        labeled_tube_cond=[(l,alg.model.intube(l.pattern)) for l in labeled]

        labeled_l,unlabeled_l=[l[0] for l in labeled_tube_cond if not l[1]],[l[0].pattern
                                                                             for l in labeled_tube_cond if l[1]]
        unlabeled_tube_cond=[(l,alg.model.intube(l)) for l in unlabeled]
        unlabeled_u,labeled_u=[l[0] for l in unlabeled_tube_cond if l[1]],[LabeledExample(l[0],alg.model.compute(l[0]))
                                                                           for l in unlabeled_tube_cond if not l[1]]
        curr_cycle+=1
        labeled=labeled_l+labeled_u
        unlabeled=unlabeled_l+unlabeled_u
    return ["e",e*len(unlabeled),res]

p=ThreadPool(2)
dataset=read_webspam_with_votelist()
res=p.map(partial(data_cleaning_experiment,dataset,5),[0.05,0.1])
for r in res:
    print r