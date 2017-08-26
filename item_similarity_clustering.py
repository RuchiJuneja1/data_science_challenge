#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 00:39:59 2017

@author: RuchiJuneja
"""


import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse
import math


# reduce text data dimensionality
def get_reduced_data(n_hasher_features, n_svd_features, data):
    hasher = HashingVectorizer(n_features=n_hasher_features, stop_words='english',
                                       norm=None, binary=False)
    vectorizer = make_pipeline(hasher, TfidfTransformer())
    X = sparse.csc_matrix(vectorizer.fit_transform(data))
    print("n_samples: %d, n_features: %d" % X.shape)
    svd = TruncatedSVD(n_svd_features)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))
    return X


# convert each category data to feature set
def get_category_data(category):
    category_data = item_listing[item_listing.category_l1_name_en == category]
    searched_data = list(category_data['item_details'].apply(lambda x:
                ' '.join([ps.stem(word.lower()) for word in 
                str(x).split() if word.lower() not in ['aed', 'eed', 'oed'] 
                and ps.stem(word.lower()) in all_keys ])).values.astype('U'))
    n_hasher_features = 15000
    n_svd_features = 1000
    item_tokens_X = get_reduced_data(n_hasher_features, n_svd_features, searched_data) 
    print("got item_tokens_X")
    category_l2_X = LabelBinarizer().fit_transform(category_data.category_l2_name_en.astype('str'))
    print("got category_l2_X")
    listing_price_X = scale(category_data['listing_price'].values.reshape(-1,1))
    print("got all")
    category_data_X = np.concatenate((item_tokens_X,category_l2_X,
                             listing_price_X), axis=1)
    return (category_data_X, category_data)


# get the best number of clusters based on highest silhouette score
def get_optimal_cluster(lower_bound, upper_bound, interval, category_data_set):            
    optimal_cluster = None
    max_silhouette_score = 0
    for k in range(lower_bound, upper_bound, interval):
        print(k)
        km = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1,
                             init_size=k, batch_size=1000)
        clusters = km.fit(category_data_set)
        sil_score = metrics.silhouette_score(category_data_set, clusters.labels_, metric='euclidean', sample_size = min(10000, category_data_set.shape[0]))
        if sil_score > max_silhouette_score:
            max_silhouette_score = sil_score
            optimal_cluster = clusters
            print(sil_score)
    return optimal_cluster


# get the tokens from user queries
stop_words = stopwords.words('english')
user_queries = pd.read_csv('/Users/RuchiJuneja/Downloads/olx_data_sample/za_queries_sample.csv')
user_queries.dropna(how='any')
dict = {}
for j in range(1,len(user_queries)):
    temp = str(user_queries['search_term'][j]).lower()
    cnt = user_queries['cnt'][j]
    tok = word_tokenize(temp)
    tok = [word for word in tok if word not in stop_words]
    for i in tok:
        if i in dict:
            pass
        else:
            dict[i] = cnt
keys = list(dict.keys())
ps = PorterStemmer()
all_keys = set([ps.stem(i.lower()) for i in dict if dict[i] > 19 ])

# load item information data and clean the dataset
item_listing = pd.read_csv('/Users/RuchiJuneja/Downloads/olx_data_sample/za_sample_listings_incl_cat.csv',
                           low_memory = False)
item_listing.dropna(how='any', inplace = True) 
item_listing = item_listing[~item_listing['seller_id'].astype(str).str.startswith('olx')]
item_listing = item_listing[~((item_listing.category_l1_name_en == "28.00690000") | 
                            (item_listing.category_l1_name_en == "28.21173000") | 
                            (item_listing.category_l1_name_en == "27.92563000") | 
                            (item_listing.category_l1_name_en == "28.04731000") | 
                            (item_listing.category_l1_name_en == "28.22927000") | 
                            (item_listing.category_l1_name_en == "18.42406000") | 
                            (item_listing.category_l1_name_en == "28.50514000") | 
                            (item_listing.category_l1_name_en == "28.00104000") | 
                            (item_listing.category_l1_name_en == "26.23637000") | 
                            (item_listing.category_l1_name_en == "28.70814000"))]   
value_counts = item_listing['category_sk'].value_counts()
to_replace = value_counts[value_counts <= 5].index                         
item_listing['category_sk'].replace(to_replace, '0', inplace=True) 

value_counts = item_listing['category_l3_name_en'].value_counts()
to_replace = value_counts[value_counts <= 5].index                         
item_listing['category_l3_name_en'].replace(to_replace, '0', inplace=True) 

item_listing['item_details'] = item_listing['listing_title'].astype(str) + " " + item_listing['listing_description'].astype(str)
item_listing['cluster_label'] = 0
    
            
# create clusters for each category separately and mark the cluster labels 
for category in (list(item_listing.category_l1_name_en.unique())):
    print(category)
    (category_dataset, raw_category_data) = get_category_data(category)
    if category_dataset.shape[0] <= 1000:
        lower_bound = int(math.ceil(category_dataset.shape[0]/100)) 
        upper_bound = int(math.ceil(category_dataset.shape[0]/10)) 
        interval = 4
    else:
        lower_bound = int(math.ceil(category_dataset.shape[0]/1000))
        upper_bound = int(math.ceil((category_dataset.shape[0]/50) / 100.0)) * 100 
        interval = 50
    optim_cluster = get_optimal_cluster(lower_bound, upper_bound, interval, category_dataset)
    category_clusters = list(optim_cluster.labels_)
    max_cluster_label = max(item_listing.cluster_label)
    category_clusters = [x + max_cluster_label for x in category_clusters]
    item_listing.loc[raw_category_data.index.tolist(),'cluster_label'] = category_clusters
  

    
    
    
    
    
    