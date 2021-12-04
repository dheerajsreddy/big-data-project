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
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LogisticRegression

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
	
	regexTokenizer = RegexTokenizer(inputCol="feature1", outputCol="words", pattern="\\W")
	# stop words remover
	
	add_stopwords = ["http","https","amp","rt","t","c","the","a", "an"] 
	stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)
	
	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
	idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=4) #minDocFreq: remove sparse terms
	
	label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
	#lr = LogisticRegression(featuresCol = 'feature1', labelCol = 'label', maxIter=10)
	#dt = DecisionTreeClassifier(featuresCol = 'feature1', labelCol = 'label')
	#rm = RandomForestClassifier(featuresCol = 'feature1', labelCol = 'label')
	#gbt = GBTClassifier(featuresCol = 'feature1', labelCol = 'label', maxIter=10)
	
	pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, nb])
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, lr])
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, dt])
	
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, rm])
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, dt])
	#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx, gbt])
	# Fit the pipeline to training documents.
	
	(trainingData, testData) = df.randomSplit([0.8, 0.2], seed = 10)

	model = pipeline.fit(trainingData)
	
	predictions = model.transform(testData)
	
	#evaluator = BinaryClassificationEvaluator(predictionCol="prediction")
	evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
	accuracy = evaluator.evaluate(predictions)
	df.show()
	
	print ("Test Area Under ROC: ", accuracy)
	

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

