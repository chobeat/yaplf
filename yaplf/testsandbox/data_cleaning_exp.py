from thesisdraw import *
from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
def data_cleaning_experiment(cycles=3):

    d=DataGenerator()
    labeled,unlabeled,a,b=read_webspam_with_votelist()
    random.shuffle(labeled)
    test_set=labeled[400:600]
    labeled=labeled[:400]
    unlabeled=unlabeled[:200]
    for c in range(cycles):
        print len(labeled),len(unlabeled)
        alg=ESVMClassificationAlgorithm(labeled,unlabeled,e=50,kernel=GaussianKernel(30),tube_tolerance=0.01)
        alg.run()

        print alg.model.quality_index(labeled,unlabeled),float(len([1 for t in test_set if alg.model.compute(t.pattern)==t.label]))/len(test_set)
        #tmp_plot(alg,labeled,unlabeled,"tmpdataplot"+str(c)+".jpg")

        labeled_tube_cond=[(l,alg.model.intube(l.pattern)) for l in labeled]

        labeled_l,unlabeled_l=[l[0] for l in labeled_tube_cond if not l[1]],[l[0].pattern
                                                                             for l in labeled_tube_cond if l[1]]
        unlabeled_tube_cond=[(l,alg.model.intube(l)) for l in unlabeled]
        unlabeled_u,labeled_u=[l[0] for l in unlabeled_tube_cond if l[1]],[LabeledExample(l[0],alg.model.compute(l[0]))
                                                                           for l in unlabeled_tube_cond if not l[1]]

        labeled=labeled_l+labeled_u
        unlabeled=unlabeled_l+unlabeled_u

data_cleaning_experiment(8)