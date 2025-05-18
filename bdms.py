from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("lin_reg").getOrCreate()
#df=spark.read.csv("/home/suryateja/spark_files/data/Sofia Air quality/*",inferSchema=True, header=True)
df = spark.read.format("csv").option("header","true").option("inferSchema","true").load("/home/suryateja/spark_files/data/Sofia Air quality/2017-09_bme280sof.csv")
df.printSchema()
print(df.head())
df =df.na.drop()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

print(df.columns)

assembler=VectorAssembler(inputCols=['pressure', 'temperature'],
                        outputCol='features')
                        
output=assembler.transform(df)

final_df=output.select('features','humidity')
train_data,test_data= final_df.randomSplit([0.7,0.3])

from pyspark.ml.regression import LinearRegression

lm=LinearRegression(featuresCol='features',labelCol='humidity',maxIter=10,regParam=0.3,elasticNetParam=0.8)
model=lm.fit(train_data)
pred=model.transform(test_data)
pred.select("prediction","humidity","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator

res=model.evaluate(test_data)
print("Root Mean Squared Error (RMSE) on test data for Linear Regression = %g" % res.rootMeanSquaredError)
print("R Squared (R2) on test data for  Linear Regression = %g" % res.r2)

from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'humidity')
dt_model = dt.fit(train_data)
dt_predictions = dt_model.transform(test_data)

dt_predictions.select("prediction","humidity","features").show(5)



dt_rmse_evaluator = RegressionEvaluator(labelCol="humidity", predictionCol="prediction", metricName="rmse")
rmse = dt_rmse_evaluator.evaluate(dt_predictions)
print("rootMeanSquaredError (rmse) on test data for Decision Tree Algorithm = %g" % rmse)

dt_r2_evaluator = RegressionEvaluator(labelCol="humidity", predictionCol="prediction", metricName="r2")
r2 = dt_r2_evaluator.evaluate(dt_predictions)
print("R Squared (R2) on test data for Decision Tree Algorithm = %g" % r2)
