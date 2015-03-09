import numpy as np
from yaplf.data import LabeledExample
from yaplf.algorithms.svm.classification import *
import warnings
import yaplf.models.kernel
import multiprocessing
from yaplf.utility.synthdataset import DataGenerator
from yaplf.testsandbox.thesisdraw import tmp_plot
from ensemble import *
from experiment_tools import *
import functools

warnings.simplefilter("error")

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



def multi_exp(params):
        training_set,unlabeled,test_set,c_i,e_i,g_i=params
        alg = ESVMClassificationAlgorithm(training_set,unlabeled,c=c_i,d=1,e=e_i*len(unlabeled),
                                             kernel=yaplf.models.kernel.GaussianKernel(g_i) ,

                                              tube_tolerance=0.0000001)
        auc=start_experiment(alg,training_set,unlabeled,test_set)
        perc=0
        index=0
        if alg.model:
            perc=float(sum([1 for t in test_set if alg.model.compute(t.pattern)==t.label]))/len(test_set)
            index=alg.model.quality_index(training_set,unlabeled)
        t=["c: "+str(c_i),"d: "+str(1),"e: "+str(e_i),"auc: "+str(auc),"perc: "+str(perc),"index: "+str(index)]
        return t

def webspam_cross_validation_experiment1():

    decided,undecided=read_dataset_temp()
    random.shuffle(decided)
    decided=decided[:1000]
    unlabeled=[i[0] for i in undecided]

    training_set=decided[:int(len(decided)*0.6)]
    test_set=decided[int(len(decided)*0.6):]
    """print "SVM Base:",cross_validation(SVMClassificationAlgorithm, yaplf.models.kernel.GaussianKernel(2),decided,10)"""
    """print "ESVM",cross_validation(ESVMClassificationAlgorithm,
                                  yaplf.models.kernel.GaussianKernel(2),
                                  decided,10,unlabeled)"""
    p=multiprocessing.Pool(3)
    res=p.map



    exps=[]

    for c_i in [5,10,20]:
        for g_i in [0.1,0.5]:
            for e_i in [0.05,0.1]:
                exps.append((training_set,unlabeled,test_set,c_i,e_i,g_i))
    res=p.map(multi_exp,exps)
    for i in res:
        print i



def webspam_weight_experiment1():

    decided,undecided=read_dataset_temp()
    unlabeled=[i[0] for i in undecided]
    l_weight,r_weight=label_to_weight([u[1] for u in undecided])
    random.shuffle(decided)

    decided=decided[:200]

    print "SVM Base:",cross_validation(SVMClassificationAlgorithm, decided,10,kernel=yaplf.models.kernel.GaussianKernel(2))
    print "ESVM",cross_validation(ESVMClassificationAlgorithm,decided,10,unlabeled,c=1,d=1,e=30,
                                    kernel=yaplf.models.kernel.GaussianKernel(2),
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
    tmp_plot(alg,decided,unlabeled)

def in_tube_variance_spam_experiment():
    import pylab as P
    decided,undecided=read_dataset_temp()
    unlabeled=[i[0] for i in undecided]
    random.shuffle(decided)

    decided=decided[:300]
    res=[]
    for e_i in [0.1,0.2,0.3,0.4,0.5,0.6]:
        e_i=len(unlabeled)*e_i/10
        alg=ESVMClassificationAlgorithm(decided,unlabeled,c=1,d=1,e=e_i,
                                    kernel=yaplf.models.kernel.GaussianKernel(2))
        try:
            alg.run()
            res.append((alg.model.tube_radius,e_i,alg.model.in_tube_ratio()))
        except Exception,err:
            print err
    if res:
        tube_radius,e_a,in_tube_a=zip(*res)
        P.plot(e_a,in_tube_a)
        P.plot(e_a,tube_radius)

        P.show()
    else:
        print "Nessun risultato"

def in_tube_variance_synthetic_experiment():
    import pylab as P
    d=DataGenerator()
    decided,unlabeled=d.generate_leap_dataset()
    res=[]
    for e_s in [0.2,0.6,1,1.4,1.8]:
        e_i=len(unlabeled)*e_s/10
        alg=ESVMClassificationAlgorithm(decided,unlabeled,c=1,d=1,e=e_i,
                                    kernel=yaplf.models.kernel.GaussianKernel(2))
        try:
            alg.run()
            res.append((alg.model.tube_radius,float(e_s)/10,alg.model.in_tube_ratio()))
            tmp_plot(alg,decided,unlabeled,"/home/chobeat/grafici/prop"+str(e_s)+".jpg")
        except Exception,err:
            print err
    if res:
        tube_radius,e_a,in_tube_a=zip(*res)
        P.plot(e_a,in_tube_a)
        P.plot(e_a,tube_radius)

        P.show()
    else:
        print "Nessun risultato"

def ensemble_experiment(ensemble_size,dataset,kernel,draw=False,ambiguousvotelist=None):
    labeled,unlabeled=dataset
    e=Ensemble(ensemble_size,labeled,unlabeled,ESVMClassificationAlgorithm,c=1000,d=1,e=10,
                                    kernel=kernel,tube_tolerance=0.00001
                                    )
    if draw:
        i=0
        for alg in e.classifiers:
            i+=1
            tmp_plot(alg,labeled,unlabeled,"/home/chobeat/grafici/ensemble"+str(i)+".jpg")

    class_votes=[e.compute(i) for i in unlabeled]

    normalized_votes=[[(x+abs(x))/2 for x in sample_vote] for sample_vote in class_votes]
    class_perc=[sum(v)/ensemble_size for v in normalized_votes]
    votelist,is_ambiguous=ambiguousvotelist
    diff=[abs(class_perc[i]-votelist[i])for i in range(len(class_perc)) if is_ambiguous[i]]

    return diff

linear_ensemble_webspam_experiment=functools.partial(ensemble_experiment,
                                                     draw=False,
                                                     kernel=LinearKernel())

gaussian_ensemble_webspam_experiment=functools.partial(ensemble_experiment,draw=False,
                                                       kernel=yaplf.models.kernel.GaussianKernel(1))
d=DataGenerator()

linear_ensemble_synth_experiment=functools.partial(ensemble_experiment,dataset=d.generate_ensemble_dataset(),draw=True,
                                                   voteList=None,kernel=LinearKernel())

gaussian_ensemble_synth_experiment=functools.partial(ensemble_experiment,dataset=d.generate_ensemble_dataset(),draw=True,
                                                    voteList=None, kernel=yaplf.models.kernel.GaussianKernel(1))


def ensemble_votes_experiment1():

    labeled,unlabeled,votelist,ambiguous=read_webspam_with_votelist()
    labeled=labeled[:2000]
    res=gaussian_ensemble_webspam_experiment(5,dataset=(labeled,unlabeled),ambiguousvotelist=(votelist,ambiguous))
    persist_result(res,"ensemble_votes1", "webspam","Formato: lista di discrepanze tra la media dei voti dell'ensemble e"
                              " l'ambiguity intrinseca nel dataset per i punti non undecided")

def main_example():

      d=DataGenerator()
      labeled,unlabeled=d.generate_simple_dataset()
      random.shuffle(labeled)

      training_set=labeled[:60]
      test_set=labeled[60:]
      for i in range(10):
            alg = ESVMClassificationAlgorithm(training_set,unlabeled,c=1,d=1,e=float(i)/70*len(training_set),
                                              kernel=yaplf.models.kernel.GaussianKernel(3),

                                              tube_tolerance=0.0001,debug_mode=False)
            path=str(home)+"/grafici/prova"+str(i)+"w.jpg"

            print start_experiment(alg,training_set,unlabeled,test_set,path)

main_example()