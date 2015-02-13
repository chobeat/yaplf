import random
import collections
class Ensemble:
    def __init__(self,size,labeled,unlabeled,algcls,*args,**kwargs):
        self.classifiers=[]

        for i in range(size):

            while(True):

                random.shuffle(labeled)
                random.shuffle(unlabeled)
                l_t=labeled[:len(labeled)/size]
                u_t=unlabeled[:len(unlabeled)/size]
                classifier=algcls(l_t,u_t,**kwargs)

                try:

                    classifier.run()
                    self.classifiers.append(classifier)
                    break
                except Exception,err:
                    print err
                    kwargs["e"]=kwargs["e"]*0.9



    def compute(self,pattern):
            votes=[classifier.model.compute(pattern) for classifier in self.classifiers]

            return  collections.Counter(votes).most_common(1)[0]
