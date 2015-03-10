__author__ = 'chobeat'

from yaplf.utility.synthdataset import *
from yaplf.models.kernel import *
from experiment_tools import read_webspam_with_votelist,evaluate_classifier,default_scaling_function
import csv
def read_iono():
    path="/home/chobeat/git/yaplf/yaplf/testsandbox/"
    keyword_map={"g":1,"b":-1}
    with open(path+"ionosphere/ionosphere.data","r") as f:
        reader=csv.reader(f)
        reader=list(reader)
        res=[LabeledExample([float(v)for v in row[:-1]],keyword_map[row[-1]])for row in reader]


    return res


