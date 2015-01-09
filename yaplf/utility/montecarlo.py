import numpy.random

class MonteCarloSimulator():
    def __init__(self):
        pass

    def OverlappingArea(self,f1,f2,c0,c1,num=100000):
        if c1<=c0:
            raise Exception("Invalid coordinates for Montecarlo evalution of overlapping areas")

        x=numpy.random.uniform(c0,c1,num)
        y=numpy.random.uniform(c0,c1,num)
        data=zip(x,y)
        count=0
        for d in data:
            a1=f1(d)
            a2=f2(d)

            if a1 != a2:
                count+=1

        overlapping=float(count)/len(data)

        return overlapping
"""
m=MonteCarloSimulator()

def fa(p):
    x,y=p
    if x>0 and y<3:
        return True
    else:
        return False

def fb(p):
    x,y=p
    if x>-1 and y<2:
        return True
    else:
        return False


m.OverlappingArea(fa,fb,-3,3)
"""