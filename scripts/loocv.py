import numpy as np
import pandas as pd
import numpy as np
from time import time
import cPickle
import pyspark
#from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import  StratifiedShuffleSplit
from sklearn.utils import shuffle
from collections import defaultdict, Counter


from pyspark.mllib.recommendation import ALS
import math

def get_ratings_data():
    # load spark ready data 
    spark_ready_data_path = "/Users/Alexander/Wine_Recommender/data/spark_ready_data.pkl"
    ratings_data = cPickle.load(open(spark_ready_data_path, "r"))
    return ratings_data

def create_cust_tag_bridge_rdd(sc, data):
    # create int:cust_tag key value pairs
    # spark can't read string user ids
    
    unique_user_tags = np.unique([row[0] for row in data])
    
    index_to_int = np.arange(0, len(unique_user_tags) * 100, 100)
    cust_tag_bridge = [ (tag_hash, tag_int) for tag_hash, tag_int in zip(unique_user_tags, index_to_int)]
    
    return sc.parallelize(cust_tag_bridge)


def create_clean_data_rdd(sc, data, cust_tag_bridge_rdd):
    # create bride rdd for customer tags and customer ids
    data_rdd = sc.parallelize(data)
    
    tag_data_bridge_rdd = data_rdd.map(lambda row: (row[0], (row[1], row[2]) ))
    
    clean_data_rdd = \
    tag_data_bridge_rdd.sortByKey()\
                   .join( cust_tag_bridge_rdd.sortByKey())\
                   .map(lambda row: ( row[1][1], row[1][0][0], row[1][0][1]))
            
    return clean_data_rdd


# split for train_test_split inside of LOOCV function  
def split_data(data_rdd):
    '''Collect data from rdd, shuffle, and split into X and Y'''
    ratings_data = data_rdd.collect()
    X = []
    Y = []
    for row in ratings_data:
        X.append([row[0], row[1]])
        Y.append(row[2])

    X, Y = shuffle(X, Y, random_state=4)
    return X, Y


# Global split - train/test sets with proportinal label distribution
# (y, n_iter=10, test_size=0.1, train_size=None, random_state=None
def get_loocv_train_test_errors(sc, X, Y , alpha, rank_ , n_iterations, file_path=None, save_file=False, return_errors=True):
    
    def predict_get_error(model, data, data_predict):
        # (r[0], r[1]), r[2]) --> user_id, wine_id, rating 
        predictions = model.predictAll(data_predict).map(lambda r: ((r[0], r[1]), r[2]))

        # combine predictions and validation sets
        rates_and_preds = data.map(lambda r: ((int(r[0]), int(r[1])), float(r[2])))\
                                        .join(predictions)
        # get RMSE for each rank
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

        rates_and_preds.unpersist()
        predictions.unpersist()
    
        return error


    # sub-sample
    initial = 0
    final = 1000000
    X, Y = X[initial:final], Y[initial:final]
    
    
    seed = 5L
    iterations = n_iterations
    regularization_parameter = alpha
    rank = rank_
    global_test_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    train_errors = []
    test_errors = []

    X = np.array(X)
    Y = np.array(Y)

    for i, test_size_ in enumerate(global_test_sizes):
        
        print "Iteration {}".format(i)

        # sss = StratifiedShuffleSplit(Y, n_iter=1, test_size=test_size_, random_state=4)
        # for train, test in sss:
        #     X_train, y_train = X[train], Y[train]
        #     X_test, y_test = X[test], Y[test]
        
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size = global_test_size,
                                                            random_state=4)
        
        train_rdd_ready = [(tuple(x)[0], tuple(x)[1], y) for x, y in zip(X_train, y_train)]
        test_rdd_ready = [(tuple(x)[0], tuple(x)[1], y) for x, y in zip(X_test, y_test)]

        train_data_rdd = sc.parallelize(train_rdd_ready)
        test_data_rdd = sc.parallelize(test_rdd_ready)
        
        model = ALS.train(
                  ratings=train_data_rdd, 
                  rank=rank, 
                  seed=seed, 
                  iterations=iterations,
                  lambda_=regularization_parameter,
                  nonnegative=True)
        
        train_data_for_predict_rdd = train_data_rdd.map(lambda row: (row[0], row[1]))
        test_data_for_predict_rdd = test_data_rdd.map(lambda row: (row[0], row[1]))

        train_error = predict_get_error(model, train_data_rdd, train_data_for_predict_rdd)
        train_errors.append(train_error)

        test_error = predict_get_error(model, test_data_rdd, test_data_for_predict_rdd)
        test_errors.append(test_error)

        # unpersist to release memory 
        train_data_rdd.unpersist()
        test_data_rdd.unpersist()
        train_data_for_predict_rdd.unpersist()
        test_data_for_predict_rdd.unpersist()

    if save_file == True:
        print "Saving results to file...."
        cPickle.dump([train_errors, test_errors], open(file_path, 'w'))

    if return_errors == True:
        return train_errors, test_errors


if __name__ == '__main__':

    loocv_path = "/Users/Alexander/Wine_Recommender/data/loocv_results_test.pkl"

    print "create sparkContext..."
    # number of nodes in local spark cluster
    n_nodes = 3
    sc = pyspark.SparkContext(master = "local[{}]".format(n_nodes))
    print "SparkContext: {}".format(sc)

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


    get_loocv_train_test_errors(sc, X, Y , alpha = 0.1, rank_ = 16, n_iterations= 20, file_path=loocv_path, save_file=True, return_errors=False)

    print "stoping spark context..."
    sc.stop()




