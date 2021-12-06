import numpy as np
import sys, pyspark, json, pickle
from sklearn import metrics
from pyspark import SparkContext
from pyspark.sql.functions import *
from sklearn.pipeline import Pipeline
from pyspark.streaming import StreamingContext
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pyspark.sql import SQLContext, SparkSession,Row,Column
from sklearn.linear_model import PassiveAggressiveClassifier
import warnings
warnings.filterwarnings("ignore")


#creating a spark context
sc = SparkContext("local[2]", "Sentiment")

#creating a streaming context to read the incoming streaming data
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

#function to convert the datastream converted to list into a dataframe object for further processing
def to_df(data):
	if data.isEmpty():
		return
	
	#creating of spark session for the streaming data input
	ss = SparkSession(data.context)
	data = data.collect()[0]
	columns = [f"feature{i}" for i in range(len(data[0]))]
	
	#create dataframe from the streaming data
	df = ss.createDataFrame(data, columns)
	df.show()	
	
	#pre-processing of data
	pre_process(df)

def pre_process(df):
	feature1 = np.array(df.select('feature1').collect())
	feature1 = feature1.flatten()
	feature0 = np.array(df.select('feature0').collect())
	feature0 = feature0.flatten()
	
	
	
	# 1. CountVectorize the data
	vect = CountVectorizer()
	X_train_counts = vect.fit_transform(feature1)
	
	# 2. Hashing TF-IDF
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	# Fit the model
	passive=PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
	clf =passive.partial_fit(X_train_tfidf, feature0,classes=np.unique(feature0))
	
	pipeline = Pipeline(([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('passive', PassiveAggressiveClassifier())]))
	pipeline.fit(feature1,feature0)
	
	with open('passive_aggr.pkl','wb') as f:
    		pickle.dump(pipeline,f)
	print("****** DONE TRAINING *******")
	
	'''
	with open('passive_aggr.pkl', 'rb') as f:
    		clf2 = pickle.load(f)
	predicted = clf2.predict(feature1)
	print(metrics.classification_report(feature0, predicted))
	
	'''
	
#function to convert the datastream to a list
def map_data(data):

	#load the incoming json file
	
	json_data=json.loads(data)
	list_rec = list()
	
	#convert the json file to tupple which is appended to a list and returned
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec.append(to_tuple)
	return list_rec 	

#creating a socket to read the data
lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)

#start streaming
ssc.start() 
ssc.awaitTermination()

