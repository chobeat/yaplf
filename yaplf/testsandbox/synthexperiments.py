__author__ = 'chobeat'
from yaplf.utility.synthdataset import *
from thesisdraw import *
from experiment_tools import cross_validation
from yaplf.models.kernel import *

def experiment_e():
    d=DataGenerator()
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    x=arange(0.1,0.6,0.05)

    exit=False
    while(not exit):
        labeled,unlabeled=d.generate_simple_dataset(50)

        y1=[]
        y2=[]
        try:
            for e_i in x:
                e=e_i*len(unlabeled)
                alg=ESVMClassificationAlgorithm(labeled,unlabeled,c=1,e=e,d=1,
                                                              kernel=GaussianKernel(0.4),
                       **kwargs)
                alg.run()

                y1.append(alg.model.tube_radius)
                y2.append(alg.model.quality_index(labeled,unlabeled))
        except Exception as e:
            print e
            continue
        exit=True
    x=map(lambda e_i:e_i*len(unlabeled),x)

    f=figure()
    p=f.add_subplot(111,xlabel='Valore di E',ylabel='Raggio del tubo')

    p.plot(x,y1)
    f.savefig("/home/chobeat/git/tesi/esperimenti/esperimentoEtubo.png")


def experiment_perf(fold=5):
    d=DataGenerator()
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}
    x=arange(0.1,0.5,0.02)

    labeled,unlabeled=d.generate_perf_dataset()
    plot_data(labeled,unlabeled,"/home/chobeat/git/tesi/esperimenti/normaldata.png")

    y1=[]

    y2=[]
    y3=[]
    for e_i in x:

            res=cross_validation(ESVMClassificationAlgorithm, labeled, fold, return_quality_index=True,
                         c=2,e=e_i*len(unlabeled),
                                                                                             d=1,
                                                      kernel=GaussianKernel(1),
                 unlabeled_sample=unlabeled,**kwargs)
            if res!=0:
                y1.append(res[0])
                y3.append(res[2])
    x=map(lambda e_i:e_i*len(unlabeled),x)
    f=figure()

    p=f.add_subplot(111,xlabel='Valore di E')
    print y1
    print y3
    print len(y1),len(y3)
    p.plot(x,y1,"r-",x,y3,"b-")
    f.savefig("/home/chobeat/git/tesi/esperimenti/esperimentoperfquality.png")


experiment_perf(5)