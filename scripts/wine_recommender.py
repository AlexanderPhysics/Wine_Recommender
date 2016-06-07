import numpy as np
import pandas as pd
import os
import cPickle
from time import time
from sklearn.utils import shuffle
from collections import defaultdict, Counter
import pyspark
from  pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
import math

def get_ratings_data(ratings_path):
    start = time()
    data = cPickle.load(open(ratings_path, 'r'))
    end = time()
    print "Time Elapsed = {:.3} seconds".format(end - start)
    return data

def create_cust_tag_bridge_rdd(sc, data):
    '''Create user tags/user ids bride rdd, 
       create int:cust_tag key value pairs,
       spark can't read string user ids'''
    
    unique_user_tags = np.unique([row[0] for row in data])

    index_to_int = np.arange(0, len(unique_user_tags) * 100, 100)
    cust_tag_bridge = [ (tag_hash, tag_int) for tag_hash, tag_int in zip(unique_user_tags, index_to_int)]

    return sc.parallelize(cust_tag_bridge)

def create_products_rdd(products_df):
    '''Creates products_rdd
    Input: products_df, pandas dataframe
    Output: products_rdd, spark rdd'''
    # create products_rdd
    products_rdd = sc.parallelize(products_df.values.tolist())
    
    # format --> (productKey, (productID, Appellation, Varietal, Vinyard) )
    products_rdd = products_rdd.map(lambda row: (row[0], (row[1], row[2], row[3], row[4], row[5]) )   )
    
    return products_rdd

def create_clean_data_rdd(data, cust_tag_bridge_rdd):
    '''Transform ratings data into spark readable format --> (user_id, productKey, rating)
    Input:  data: list, cust_tag_bridge_rdd: spark rdd
    Output: clean_data_rdd, spark rdd'''
    data_rdd = sc.parallelize(data)
    
    tag_data_bridge_rdd = data_rdd.map(lambda row: (row[0], (row[1], row[2]) ))
    
    clean_data_rdd = \
    tag_data_bridge_rdd.sortByKey()\
                   .join( cust_tag_bridge_rdd.sortByKey())\
                   .map(lambda row: ( row[1][1], row[1][0][0], row[1][0][1]))
            
    return clean_data_rdd

def get_spark_context(n_cups = 3, local = True, remote_cluster_path=None):
    # number of nodes in local spark cluster
    n_worker_cups = n_cups
    if local == True:
        print "Create spark context for local cluster..."
        sc = pyspark.SparkContext(master = "local[{}]".format(n_worker_cups))
        return sc
    elif local == False:
        print "Create spark context for remote cluster..."
        sc = pyspark.SparkContext(master = remote_cluster_path)
        return 
    else:
        print "ERROR: local is set to False, however remote_cluster_path is not specified!"

def get_clean_data_rdd(sc, return_cust_brige_rdd = False):
    '''Loads ratings from master file and formats data into model readable form.
       data --> (user_id, productKey, rating)'''
    # load data
    data = get_ratings_data(ratings_path)
    # assigne each user hash tag a user_id
    cust_tag_bridge_rdd = create_cust_tag_bridge_rdd(sc, data)
    # model readable format
    clean_data_rdd = create_clean_data_rdd(data, cust_tag_bridge_rdd) 
    
    if return_cust_brige_rdd == False:
        cust_tag_bridge_rdd.unpersist()
        return clean_data_rdd
    else:
        return clean_data_rdd, cust_tag_bridge_rdd

def train_model(training_RDD):
    # TODO: still need to optimize hyperparameters in a grid search
    seed = 5L
    iterations = 30
    regularization_parameter = 0.1
    rank = 20

    model = ALS.train(training_RDD, 
                      rank=rank, 
                      seed=seed, 
                      iterations=iterations,
                      lambda_=regularization_parameter,
                      nonnegative=True)
    return model

def get_trained_model(sc, ratings_path, save_model_path=None, return_clean_data_rdd=False):
    '''Loads rating data from file, trains model, and returns a fitted model'''
    
    print "load data and build RDDs..."
    clean_data_rdd = get_clean_data_rdd(sc, return_cust_brige_rdd = False)
    
    print "Training Model..."
    start = time()
    fitted_model = train_model(clean_data_rdd )
    end = time()
    print "Training Model: Time Elapsed = {:.3} \n".format(end - start)
    


    if save_model_path != None:
        # Save model
        print "saving model to path: {}".format(save_model_path)
        fitted_model.save(sc ,save_model_path)

    if return_clean_data_rdd:
        return fitted_model, clean_data_rdd
    else:
        # restore memory resources
        clean_data_rdd.unpersist()
        return fitted_model


def load_model(sc, model_path):
    '''Load trained model that has been saved to file. 
       It is more efficient to train a model once, then make predictions.'''
    # load model
    fitted_model = MatrixFactorizationModel.load(sc, model_path)
    return fitted_model


def get_userID_moiveID_pairs(sc, user_id, clean_data_rdd):
    '''In order to get recommendations for a user, we need to build an RDD with (user_id, wine_id)
       pairs for wines that the user has not previously purchased.'''
    # ( user_id, movie_id, rating  )
    # get user_id's  movie ids in a list
    movie_ids = clean_data_rdd.filter(lambda row: row[0] == user_id )\
                              .map(lambda row: row[1]).collect()
        
    # get wine_ids that user_id has not purchased 
    unpurchased_wines = clean_data_rdd.filter(lambda row: row[0] != user_id )\
                                      .filter(lambda row: row[2] not in  movie_ids)\
                                      .map(lambda row: (user_id, row[1] ) ).distinct()
    return unpurchased_wines

def get_user_recommendations(fitted_model, unpurchased_wines):
    user_recs = fitted_model.predictAll(unpurchased_wines)
    return user_recs

def format_user_recs(user_recs, cust_tag_bridge_rdd, products_path, thresh ):
    '''Reformat user recommendations so it's human readable and in preperation for curation.
       This function swaps the user_id back to the original user hash tag, and attachs the wine
       features (i.e. productID, appellation, varieatl, ...) '''
    
    # value validated in Spark_Recommendation_Model_Validation notebook
    threshold = thresh
    validated_user_recs = user_recs.filter(lambda row: row[2] >= threshold )
    
    # format --> (product key, predicted rating, user hash tag)
    wineID_rating_userHash = \
    validated_user_recs.map(lambda row:  (row[0], (row[1], row[2]) )  )\
                       .join(cust_tag_bridge_rdd\
                       .map(lambda row: (row[1], row[0])))\
                       .map(lambda row: (row[1][0][0],
                                        (row[1][0][1],
                                         row[1][1] ) ))  

    products_df = pd.read_pickle(products_path)                  
    products_rdd = create_products_rdd(products_df)
    # Key:Value pair RDD
    # format --> (custumer tag, (productKey , productID, Appellation, Varietal, Vineyard, wine type, Rating  ) )  
    clean_user_recs = \
    wineID_rating_userHash.join(products_rdd)\
                          .map(lambda row: ( row[1][0][1], 
                                             (row[0], 
                                              row[1][1][0], 
                                              row[1][1][1], 
                                              row[1][1][2], 
                                              row[1][1][3],
                                              row[1][1][4],
                                              row[1][0][0])))
    return clean_user_recs

def curate_top_wines(top_varietal_recs, top_varietals):
    final_recs = defaultdict(list)
    for var in top_varietals:
        var_cnt = 1
        for row in top_varietal_recs:
            if row[1][3] == var:
                if var_cnt <= 3:
                    var_cnt += 1
                    #final_recs.append((row[0], row[1][:-1]))
                    final_recs[row[0]].append(row[1][:-1])
    return final_recs

def get_top_rec_varietals(clean_user_recs):
    '''Returns the top 3 wines from the top 3 varietals for user'''
    
    # { custumer tag : (productKey , productID, Appellation, Varietal, Vineyard, wine type, Rating  ) }
    user_recs_dicts = clean_user_recs.collect()
    varietals = [row[1][3] for row in user_recs_dicts]
    
    var_count = Counter(varietals)
    
    # get top 3 most recommender varietals for this user
    top_varietals =  [row[0] for row in var_count.most_common()[0:3]] 
    
    top_varietal_recs = clean_user_recs.filter(lambda row: row[1][3] in  top_varietals ).collect()
    
    return curate_top_wines(top_varietal_recs, top_varietals)

def get_top_reds_and_whites(clean_user_recs):
    '''Returns top rated wines, 5 red and 5 white for user'''

    # { custumer tag : (productKey , productID, Appellation, Varietal, Vineyard, wine type, Rating  ) }
    user_recs_dicts = clean_user_recs.collect()
    
    red_white_recs_dict = defaultdict(list)
    white_cnt = 1
    red_cnt = 1
    for rec in user_recs_dicts:
        if rec[1][5] == "White Wines":
            if white_cnt <= 5:
                red_white_recs_dict[rec[0]].append(rec[1])
                white_cnt += 1
        else:
            if red_cnt <= 5:
                red_white_recs_dict[rec[0]].append(rec[1])
                red_cnt += 1
                
    return red_white_recs_dict

def get_user_ids_for_recommendations(cust_tag_bridge_rdd):
    '''This function returns user ids from the cust_tag_bridge_rdd. 
       For now, it only return the first user_id in the rdd.'''
    # results are inside of a list 
    return cust_tag_bridge_rdd.map(lambda row: row[1]).collect()

def check_top_varietal_wine_count(most_common_varietals):
    '''Checks if top variatls have at lease 3 wines'''
    cnt = 0
    for row in most_common_varietals:
        if row[1] >= 3:
            cnt += 1
    return cnt


if __name__ == '__main__':

    start_rs = time()
    # data files
    home = "/Users/Alexander/Wine_Recommender/data/"
    ratings_path = home + "spark_ready_data.pkl"
    products_path = home + "wine_products.pkl"
    rec_results_path = home + "user_rec_results.pkl"
    # trained recommender path 
    model_path = "/Users/Alexander/Wine_Recommender/models/spark_recommender"
    n_local_cpus = 3
    # value validated in Spark_Recommendation_Model_Validation notebook
    rating_threshold = 7
    n_varietials = 3

    print "get_spark_context..."
    # get sparkContext 
    sc = get_spark_context(n_cups = n_local_cpus, 
                           local = True, 
                           remote_cluster_path=None)

    print "get_clean_data_rdd..."
    clean_data_rdd, cust_tag_bridge_rdd = get_clean_data_rdd(sc, 
                                                             return_cust_brige_rdd = True)

    print 'get_trained_model...'
    # Model can be saved to a file only once; otherwise, spark will throw an error 
    fitted_model = get_trained_model(sc, 
                                     ratings_path, 
                                     save_model_path=model_path)

    # print "load_model..."
    # fitted_model = load_model(sc, 
    #                           model_path)

 

    print "get_user_ids_for_recommendations..."
    user_ids = get_user_ids_for_recommendations(cust_tag_bridge_rdd)

    r_w_cnt = 0
    results = []
    for i, user_id in enumerate(user_ids[0:3]):

        loop_start = time()
        #all previously unpurchased wines will be passed into the model for a predicted rating 
        #print "get_userID_moiveID_pairs..."
        unpurchased_wines = get_userID_moiveID_pairs(sc, 
                                                 user_id, 
                                                 clean_data_rdd)

        #print "get_user_recommendations..."
        user_recs = get_user_recommendations(fitted_model, 
                                         unpurchased_wines)

        clean_user_recs = format_user_recs(user_recs, 
                                       cust_tag_bridge_rdd, 
                                       products_path, 
                                       rating_threshold)

        # Curate Recommendations into Varietal Sub-Genres
        # Return the top 3 rated wines from the the top 3 most recommended varietals. 
        # If there aren't at least 3 wines form 3 varietals,
        # Then return the top 5 reds and the top 5 whitesn (though this shouldn't be a problem). 

        # check for 3 wines, 3 varieatls condition

        # format -> (custumer tag,  (productKey , productID, Appellation, Varietal, Vineyard, wine type, Rating  ) )
        user_recs_tups = clean_user_recs.collect()
        varietals = [row[1][3] for row in user_recs_tups]
        var_count = Counter(varietals)
        most_common_varietals = var_count.most_common()[:n_varietials]

        # check 1 -->  varietal count
        # check 2 --> top 3 varietals have at least 3 wines to choose from
        if len(var_count) >= n_varietials and check_top_varietal_wine_count(most_common_varietals) == n_varietials:
            #print "get_top_rec_varietals..."
            final_recs = get_top_rec_varietals(clean_user_recs)
        else:
            #print "get_top_reds_and_whites..."
            r_w_cnt += 1
            final_recs = get_top_reds_and_whites(clean_user_recs)
        results.append(final_recs)

        if i % 1 == 0:
            loop_end = time()
            print "User {}, Time Elapsed {:.3} mins".format(i, (loop_end - loop_start)/60)

    print "saving final_recs to file..."
    # save recommendation results to file
    cPickle.dump(results, open(rec_results_path, 'w'))


    print  "stoping spark context..."
    sc.stop()
    end_rc = time()
    print "Red_White_Rec_Counter = {}".format(r_w_cnt)
    print "Total Time Elapsed for RS = {:.4} mins".format((end_rc - start_rs)/60)






