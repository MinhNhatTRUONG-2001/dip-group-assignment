// Databricks notebook source
// MAGIC %md
// MAGIC Copyright 2025 Tampere University<br>
// MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
// MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
// MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

// COMMAND ----------

// imports for the entire notebook
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}

// COMMAND ----------

// MAGIC %md
// MAGIC This advanced task involves experimenting with the classifiers provided by the Spark machine learning library. Time series data collected in the ProCem research project is used as the training and test data. Similar data in a slightly different format was used in the last tasks of the weekly exercise 3.
// MAGIC
// MAGIC The folder `assignment/kampusareena` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains measurements from Hervanta campus.
// MAGIC
// MAGIC The dataset is given in Parquet format, and it contains data from a period of 6 months, from May 2025 to October 2025.<br>
// MAGIC Each row contains the average of the measured values for a single minute. The following columns are included in the data:
// MAGIC
// MAGIC | column name        | column type   | description |
// MAGIC | ------------------ | ------------- | ----------- |
// MAGIC | timestamp          | timestamp     | The timestamp for this row's measurements |
// MAGIC | temperature        | double        | The temperature measured by the weather station on top of Sähkötalo (`°C`) |
// MAGIC | humidity           | double        | The humidity measured by the weather station on top of Sähkötalo (`%`) |
// MAGIC | power_water_cooling_01 | double    | The electricity power consumed by the first water cooling machine on Kampusareena (`W`) |
// MAGIC | power_water_cooling_02 | double    | The electricity power consumed by the second water cooling machine on Kampusareena (`W`) |
// MAGIC | power_ventilation  | double        | The electricity power consumed by the ventilation machinery on Kampusareena (`W`) |
// MAGIC | power_elevator_01  | double        | The electricity power consumed by the first elevator on Kampusareena (`W`) |
// MAGIC | power_elevator_02  | double        | The electricity power consumed by the second elevator on Kampusareena (`W`) |
// MAGIC | power_ev_charging  | double        | The electricity power consumed by the electric vehicle charging station on Kampusareena (`W`) |
// MAGIC | power_solar_plant  | double        | The total electricity power produced by the solar panels on Kampusareena (`W`) |
// MAGIC
// MAGIC #### General guide for each case in advanced task 4
// MAGIC
// MAGIC - Load the data from the storage to a data frame. (this needs to be done only for the first case, the same data can be reused in later cases)
// MAGIC - Calculate any values that are not yet explicitly available, but are needed for the case.
// MAGIC - Clean the data and remove any rows that contain missing values (i.e., null values), in the columns that are needed for the case.
// MAGIC - Split the dataset into training and test parts.
// MAGIC - Train a machine learning model using a [Random forest classifier](https://spark.apache.org/docs/3.5.6/ml-classification-regression.html#random-forests) with the case-specific inputs and labels.
// MAGIC - Evaluate the accuracy of the trained model using the test part of the dataset according to the case-specific instructions.
// MAGIC
// MAGIC In all cases, you are free to choose the training parameters as you wish. However, don't pick parameters that make the training take a very long time (even if it would produce a more accurate model).<br>
// MAGIC Also, note that it is advisable that while you are building your task code to only use a portion of the full 6-month dataset in the initial experiments.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Advanced Task 4 - Case 2 - Predicting whether it is a weekend or not
// MAGIC
// MAGIC - Train a model to predict whether it is a **weekend** (Saturday or Sunday) or a weekday (Monday-Friday) based on five power values:
// MAGIC     - the total water cooling machine power consumption, i.e., the sum of the power consumptions values for the two water cooling machines
// MAGIC     - the ventilation machine power consumption
// MAGIC     - the total elevator power consumption, i.e., the sum of the power consumption values for the two elevators
// MAGIC     - the electric vehicle charging station power consumption
// MAGIC     - the power production value for the solar panels
// MAGIC - Evaluate the accuracy of the trained model by calculating the accuracy percentage, i.e., how often it predicts the correct value, and by calculating how accurate the prediction is for each separate day of the week.
// MAGIC     - For the accuracy measurement, you can use the Spark built-in multi-class classification evaluator, or calculate it by yourself using the prediction data frame.
// MAGIC     - For the separate day of the week accuracy, calculate the accuracy for predictions where the actual day was Monday, and the same for Tuesday, ...

// COMMAND ----------

// Preprocess and split data

val procemDayOfWeekDF: DataFrame = procemDF.na.drop
  .select(col("timestamp"), (col("power_water_cooling_01") + col("power_water_cooling_02")).as("power_water_cooling"), col("power_ventilation"), (col("power_elevator_01") + col("power_elevator_02")).as("power_elevator"), col("power_ev_charging"), col("power_solar_plant"))
  .withColumn("weekday_int", weekday(col("timestamp")))
  .withColumn("weekday", date_format(col("timestamp"), "EEEE"))
  .withColumn("label", when(col("weekday").isin("Saturday", "Sunday"), lit("weekend")).otherwise(lit("weekday")))
  .drop("timestamp")

val Array(procemCase2TrainingDataDF, procemCase2TestDataDF) = procemDayOfWeekDF.randomSplit(Array(0.8, 0.2), seed=123)


// COMMAND ----------

// Create pipeline and train model

val case2VectorAssembler = new VectorAssembler()
  .setInputCols(Array("power_water_cooling", "power_ventilation", "power_elevator", "power_ev_charging", "power_solar_plant"))
  .setOutputCol("features")

val case2LabelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(procemCase2TrainingDataDF)

val case2RandomForestClassifier = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("features")
  .setNumTrees(5)
  .setMaxDepth(10)
  .setSeed(123)

val case2LabelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(case2LabelIndexer.labelsArray(0))

val case2Pipeline = new Pipeline()
  .setStages(Array(case2VectorAssembler, case2LabelIndexer, case2RandomForestClassifier, case2LabelConverter))

val case2Model = case2Pipeline.fit(procemCase2TrainingDataDF)

// COMMAND ----------

// Test the model
val case2Predictions = case2Model.transform(procemCase2TestDataDF)

// Evaluate the model
val case2Evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setRawPredictionCol("rawPrediction")

val case2Accuracy: Double = case2Evaluator.evaluate(case2Predictions)

val case2AccuracyDF = case2Predictions
  .groupBy("weekday_int", "weekday")
  .agg(
    (round(sum(when(col("prediction") === col("indexedLabel"), 1).otherwise(0)) / count("*") * 100, 2)).as("accuracy")
  )
  .orderBy("weekday_int")
  .drop("weekday_int")

// COMMAND ----------

println(s"The overall accuracy of the weekend prediction model is ${scala.math.round(case2Accuracy*10000)/100.0} %")
println("Accuracy (in percentages) of the weekend predictions based on the day of the week:")
case2AccuracyDF.show(false)