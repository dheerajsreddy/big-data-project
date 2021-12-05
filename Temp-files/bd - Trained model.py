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
	#split = df.randomSplit([0.6,0.4]) 
	#train_df=split[0]
	mvv_list=df.select('feature1').collect()
	mvv_array = [row.feature1 for row in mvv_list]

	stage_1 = RegexTokenizer(inputCol= 'feature1' , outputCol= 'regex_done')
	stage_2 = StopWordsRemover(inputCol= 'regex_done', outputCol= 'filtered_words')
	stage_3 = Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
	model = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="vector" ,labelCol="feature0")
	pipeline = Pipeline(stages= [stage_1,stage_2,stage_3,model])
	#pipelineFit = pipeline.fit(df)

	classifier = pipeline.fit()
	
	

def map_data(data):
	json_data=json.loads(data)
	list_rec = list()
	for rec in json_data:
		to_tuple = tuple(json_data[rec].values())
		list_rec .append(to_tuple)
	return list_rec 	

lines = ssc.socketTextStream("localhost",6100).map(map_data).foreachRDD(to_df)

ssc.start() 
ssc.awaitTermination(50)
ssc.stop()
