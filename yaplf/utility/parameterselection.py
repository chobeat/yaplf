import sklearn.grid_search
from yaplf.data import LabeledExample
import random
import yaplf.models.kernel
from yaplf.utility.synthdataset import DataGenerator
from yaplf.algorithms.svm.classification import *
from yaplf.testsandbox.experiment_tools import *
class ClassifierBridge():

    def __init__(self,cls,args,kwargs,evaluation_func):
        self.args=args
        self.kwargs=kwargs
        self.cls=cls
        self.evaluation_func=evaluation_func

    def fit(self,x,y):

        if self.cls==SVMClassificationAlgorithm:
            self.c=1
            self.d=1
            self.e=1

        self.alg=self.cls([LabeledExample([1,1],1)],*self.args,c=self.c,d=self.d,e=self.e,
                          kernel=self.kernel,dummy=True,**self.kwargs)
        labeled=[LabeledExample(x[i],y[i]) for i in range(len(x)) ]
        self.alg.sample=labeled
        try:
            self.alg.run()
        except Exception,err:
            print err

    def predict(self,x):
        if self.alg.model:
            return self.alg.model.compute(x)
        else:
            return 0
    def score(self,x,y):
        test=[LabeledExample(x[i],y[i]) for i in range(len(x)) ]
        if self.alg.model:

            score=self.evaluation_func(self.alg.model,test,
                                       scaling_function=logistic_scaling_function,scaling_params=[1])

            return score
        else:
            return 0
    def get_params(self,deep=True):
       return {"cls":self.cls,"args":self.args,"kwargs":self.kwargs,"evaluation_func":self.evaluation_func}

    def set_params(self,**params):
        for name,param in params.iteritems():
            setattr(self,name,param)


def param_search(cls,labeled,params_list,c_args,c_kwargs,evaluation_func=evaluate_classifier):
    estimator=ClassifierBridge(cls,c_args,c_kwargs,evaluation_func=evaluation_func)
    gs=sklearn.grid_search.GridSearchCV(estimator,params_list,refit=False)
    X=[x.pattern for x in labeled]
    Y=[x.label for x in labeled]
    gs.fit(X,Y)
    return gs.grid_scores_

def param_to_kernel(k,param_list):
    return map(lambda x:k(x),param_list)


if __name__=="__main__":
    d=DataGenerator()
    labeled,unlabeled,a,b=read_webspam_with_votelist()
    search_kernel=yaplf.models.kernel.GaussianKernel
    random.shuffle(labeled)
    pos,neg=split_by_label(labeled)
    labeled=pos[:200]+neg[:300]
    kwargs= {"tube_tolerance":0.01,"debug_mode":False}


    res0=param_search(ESVMClassificationAlgorithm,labeled,{"c":[4,10],
                                                    "e":[0.20*len(unlabeled),0.25*len(unlabeled)],"d":[10,20],
                                                    "kernel":param_to_kernel(search_kernel,[2])}
                 ,c_args=[unlabeled],c_kwargs=kwargs)

    res1=param_search(SVMClassificationAlgorithm,labeled,{"c":[1],"e":[1],"d":[1],
                                                      "kernel":param_to_kernel(search_kernel,[1])}
                 ,c_args=[unlabeled],c_kwargs=kwargs)

    print res0,res1
