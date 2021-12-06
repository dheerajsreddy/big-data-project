import numpy as np
import sys, pyspark, json
from sklearn import metrics
from pyspark import SparkContext
from pyspark.sql.functions import *
from sklearn.pipeline import Pipeline
from pyspark.streaming import StreamingContext
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pyspark.sql import SQLContext, SparkSession,Row,Column

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

	#train, test split of data
	train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(feature1, feature0, test_size=0.20, random_state=42)

	# 1. CountVectorize the data
	vect = CountVectorizer()
	X_train_counts = vect.fit_transform(feature1)

	# 2. Hashing TF-IDF
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

	# Fit the model

	#clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
	#scaler = StandardScaler(feature1)
	#le = LabelEncoder()
	#x_train = le.fit(feature1)
	
	clf = SGDClassifier().partial_fit(X_train_tfidf, feature0, classes=np.unique(feature0))
	#clf.fit(train_data_x,train_data_y)
	
	
	pipeline=Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('sgdclassifier', SGDClassifier())])
	#pipeline=Pipeline(steps=[('scaler',StandardScaler()),('sgdclassifier', SGDClassifier())])
	pipeline.fit(train_data_x,train_data_y)

	predicted = pipeline.predict(test_data_x)
	print(metrics.classification_report(test_data_y, predicted))
	

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
