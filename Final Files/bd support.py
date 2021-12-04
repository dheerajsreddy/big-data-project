****************************** BETTER THAN ROC PLOTTED - 2, MORE THAN 0.60 ACCURACY CONSISTENTLY ******************************************************************************************


import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext, SparkSession,Row,Column
from pyspark.sql.functions import *
from pyspark.streaming import StreamingContext
#from pyspark.sql.DataFrame import  randomSplit
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, VectorAssembler,Tokenizer, OneHotEncoder

#creating a spark context
sc = SparkContext("local[2]", "Sentiment")

#creating a streaming context to read the incoming streaming data
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

'''
def pre_process(df):
	#to clean and preprocess the data
	
	print("IN")
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
'''

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
	
	print("IN")
	
	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="words", pattern="\\W")
	# stop words
	add_stopwords = ["http","https","amp","rt","t","c","the"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	# bag of words count
	#countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=90000, minDF=5)
	
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4) #minDocFreq: remove sparse terms
	
	label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx, nb])
	
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, nb])
	
	# Fit the pipeline to training documents.
	(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 100)

	lrModel = pipeline.fit(trainingData)
	
	predictions = lrModel.transform(testData)
	
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	
	print ("Test Area Under ROC: ", accuracy)
	
	'''
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
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




****************************** BETTER THAN ROC PLOTTED - 1, MORE THAN 0.55 ACCURACY CONSISTENTLY ******************************************************************************************
#FOR 1000 IT IS EVEN BETTER
 


import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext, SparkSession,Row,Column
from pyspark.sql.functions import *
from pyspark.streaming import StreamingContext
#from pyspark.sql.DataFrame import  randomSplit
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, VectorAssembler,Tokenizer, OneHotEncoder

#creating a spark context
sc = SparkContext("local[2]", "Sentiment")

#creating a streaming context to read the incoming streaming data
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

'''
def pre_process(df):
	#to clean and preprocess the data
	
	print("IN")
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
'''

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
	
	print("IN")
	
	# regular expression tokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="words", pattern="\\W")
	# stop words
	add_stopwords = ["http","https","amp","rt","t","c","the"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	# bag of words count
	countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=90000, minDF=5)
	
	#hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	#idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4) #minDocFreq: remove sparse terms
	
	label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx, nb])
	
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, nb])
	
	# Fit the pipeline to training documents.
	(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 100)

	lrModel = pipeline.fit(trainingData)
	
	predictions = lrModel.transform(testData)
	
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	
	print ("Test Area Under ROC: ", accuracy)
	
	'''
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
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

****************************** BETTER THAN ROC PLOTTED - 1, MORE THAN 0.55 ACCURACY CONSISTENTLY ******************************************************************************************

****************************** BETTER THAN ROC PLOTTED - 1, MORE THAN 0.55 ACCURACY CONSISTENTLY ******************************************************************************************

****************************** ROC PLOTTED - 1 ******************************************************************************************
import sys, pyspark, json 
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext, SparkSession,Row,Column
from pyspark.sql.functions import *
from pyspark.streaming import StreamingContext
#from pyspark.sql.DataFrame import  randomSplit
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer, VectorAssembler,Tokenizer, RegexTokenizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#creating a spark context
sc = SparkContext("local[2]", "Sentiment")

#creating a streaming context to read the incoming streaming data
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

'''
def pre_process(df):
	#to clean and preprocess the data
	
	print("IN")
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")
'''

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
	
	print("IN")
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")

	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)

	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")

	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")

	train, test = df.randomSplit([0.7, 0.3])
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

	pipeline = Pipeline(stages=[regexTokenizer, cv, indexer, vecAssembler,nb])

	model = pipeline.fit(train)
	
	predictions = model.transform(test)
	
	# Select results to view
	predictions.limit(10).select("label", "prediction").show(truncate=False)

	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Test Area Under ROC: ", accuracy)
	print("OUT")

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


****************************** ROC PLOTTED - 1 ******************************************************************************************

****************************** ROC PLOTTED - 1 ******************************************************************************************



import sys, pyspark, json
from pyspark import SparkContext
from pyspark.ml import Pipeline
import pyspark.sql.types as tp
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SQLContext, SparkSession,Row,Column
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import VectorAssembler, StringIndexer 
from pyspark.ml.classification import NaiveBayes 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

sc = SparkContext("local[2]", "Sentiment")
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

def to_df(data):
	if data.isEmpty():
		return
		
	ss = SparkSession(data.context)
	data = data.collect()[0]
	columns = [f"feature{i}" for i in range(len(data[0]))]
	
	df = ss.createDataFrame(data, columns)
	
	'''
	split = df.randomSplit([0.6,0.4]) 
	train_df=split[0]
	train_test_df = split[1]

	#X = train_df.select('feature1')
	#y = train_test_df.select('feature0') 
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
	'''

	mvv_list=df.select('feature1').collect()
	mvv_array = [row.feature1 for row in mvv_list]

	'''
	
	count_vect = CountVectorizer()
	tfidf_transformer = TfidfTransformer()
		
	X_train_counts = count_vect.fit_transform(df.select('feature1'))
	#X_train_counts = count_vect.fit_transform(df.select('filtered_words'))
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	clf = MultinomialNB().fit(X_train_tfidf, train_data.target)

	#stage_1 = RegexTokenizer(inputCol= 'feature1' , outputCol= 'regex_done')
	#stage_2 = StopWordsRemover(inputCol= 'regex_done', outputCol= 'filtered_words')
	#stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	#model = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="vector" ,labelCol="feature0")

	pipeline = Pipeline(stages= [('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
	classifier = pipeline.fit(df.select('feature1'),df.target)
	#classifier = pipeline.fit(df)
	'''
	model = make_pipeline(TfidfVectorizer(), MultinomialNB())
	# Train the model using the training data
	model.fit(df.select('feature1'), df.select('feature0'))

	
	'''
	train_pred = classifier.predict(train_test_df)
	
	ac = accuracy_score(train_test_df.select('feature0'),train_pred)
	cm = confusion_matrix(train_test_df.select('feature0'), train_pred)
	print(ac)
	print(cm)
	'''
def preprocess(df):

	stages = []
	
	# 1. clean data and tokenize sentences using RegexTokenizer
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="tokens", pattern="\\W+")
	stages += [regexTokenizer]
	
	# 2. CountVectorize the data
	cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)
	stages += [cv]
	
	# 3. Convert the labels to numerical values using binariser
	indexer = StringIndexer(inputCol="feature0", outputCol="label")
	stages += [indexer]
	
	# 4. Vectorise features using vectorassembler
	vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
	stages += [vecAssembler]
	
	pipeline = Pipeline(stages=stages)
	data = pipeline.fit(df).transform(df)
	
	train, test = data.randomSplit([0.7, 0.3])
	
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	# Fit the model
	model = nb.fit(train)

	'''
	predictions = model.transform(test)
	predictions.select("label", "prediction").show()
	
	
	evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	print ("Model Accuracy: ", accuracy)
	'''
	

def map_data(data):
	json_data=json.loads(data)
	list_rec = list()
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec.append(to_tuple)
	return list_rec 	

lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)

ssc.start() 
ssc.awaitTermination()
ssc.stop()
