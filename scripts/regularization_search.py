import numpy as np
import pandas as pd
import numpy as np
from time import time
import cPickle
import pyspark
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.utils import shuffle
from collections import defaultdict, Counter
from pyspark.mllib.recommendation import ALS
import math


from loocv import get_ratings_data, create_cust_tag_bridge_rdd, create_clean_data_rdd, split_data, get_loocv_train_test_errors

def get_regularization_results(sc, X, Y , grid_search_path, rank, n_iterations, reg_parameters):
    train_errors_dict = dict()
    test_errors_dict = dict()
    for param in reg_parameters:
        print "Starting {}".format(param)
        train_errors, test_errors = get_loocv_train_test_errors(sc, X, Y , param, rank, n_iterations, file_path=grid_search_path, save_file=False)
        train_errors_dict[param] = train_errors
        test_errors_dict[param] = test_errors

    print "Saving Results to File..."
    cPickle.dump([train_errors_dict, test_errors_dict], open(grid_search_path, 'w'))


if __name__ == '__main__':

    grid_search_path = "/Users/Alexander/Wine_Recommender/data/regularization_search_results.pkl"

    print "create sparkContext..."
    # number of nodes in local spark cluster
    n_nodes = 3
    sc = pyspark.SparkContext(master = "local[{}]".format(n_nodes))
    print "SparkContext: {}".format(sc)

    reg_parameters = [1.0, 0.1, 0.01, 0.001]
    rank=16
    n_iterations=20

    # load data
    print "load data..."
    data = get_ratings_data()
    print "build RDDs..."
    cust_tag_bridge_rdd = create_cust_tag_bridge_rdd(sc, data)
    clean_data_rdd = create_clean_data_rdd(sc, data, cust_tag_bridge_rdd) 
    X, Y = split_data(clean_data_rdd)

    print "unpersisting initial data RDDs..."
    cust_tag_bridge_rdd.unpersist()
    clean_data_rdd.unpersist()


    print "Start get_regularization_results..."
    start = time()
    get_regularization_results(sc, X, Y , grid_search_path, rank, n_iterations, reg_parameters)
    end = time()
    print "Time Elapsed = {}".format(end - start)

    print "stoping spark context..."
    sc.stop()





