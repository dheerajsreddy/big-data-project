import numpy as np
import sys, pyspark, json, pickle
from sklearn import metrics
from pyspark import SparkContext
from pyspark.sql.functions import *
from sklearn.pipeline import Pipeline
#from pyspark.ml import Pipeline
#from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StringIndexer, CountVectorizer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.streaming import StreamingContext
from sklearn.naive_bayes import MultinomialNB
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
	model_build(df)
	#model_test_build(df)

'''
def model_test_build(df):

	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")
	
	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)
	
	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
	vecAssembler = vecAssembler["features"].reshape(-1,1)
	
	
	df.show()
	# Fit the model
	#nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	clf = MultinomialNB().fit(vecAssembler, indexer)
	#clf = MultinomialNB().partial_fit(vecAssembler, indexer)
	
	pipeline = Pipeline(stages=[regexTokenizer, cv, vecAssembler, indexer, clf])
	
	(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 10)
	
	model = pipeline.fit(trainingData)
	
'''	

def model_build(df):

	feature1 = np.array(df.select('feature1').collect())
	feature1 = feature1.flatten()
	
	feature0 = np.array(df.select('feature0').collect())
	feature0 = feature0.flatten()
	
	#train, test split of data
	file_name1='naive_bayes.sav'
	train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(feature1, feature0, test_size=0.00001, random_state=42)
	
	# 1. CountVectorize the data
	vect = CountVectorizer()
	X_train_counts = vect.fit_transform(feature1)
	
	# 2. Hashing TF-IDF
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	
	# Fit the model
	clf1 = MultinomialNB()
	clf = MultinomialNB().partial_fit(X_train_tfidf, feature0, classes=np.unique(feature0))
	
	#pipelining of all the stages that each batch has to go through 
	pipeline = Pipeline(([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]))
	model = pipeline.fit(train_data_x,train_data_y)
	
	pickle.dump(clf1,open(file_name1,'wb'))
	print("****** DONE TRAINING *******")
	
	'''
		#testing
	clf1=pickle.load(open(file_name1,'rb'))
	predicted = clf1.predict(test_data_x)
	print(metrics.classification_report(test_data_y, predicted))
	#evaluating predictions on a test dataset and accuracy of the model
	#predicted = pipeline.predict(test_data_x)
	#print(metrics.classification_report(test_data_y, predicted))
	#print(metrics.accuracy_score(test_data_y, predicted))
	'''
	
	
def model_test(df):

	model=model.load("/home")
	
	feature1 = np.array(df.select('feature1').collect())
	feature1 = feature1.flatten()
	
	predicted = model.predict(feature1)
	
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

