// Databricks notebook source
// MAGIC %md
// MAGIC Copyright 2025 Tampere University<br>
// MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
// MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
// MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

// COMMAND ----------

// imports for the entire notebook
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// COMMAND ----------

// MAGIC %md
// MAGIC ## Advanced Task 2 - Wikipedia articles
// MAGIC
// MAGIC The folder `assignment/wikipedia` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains a number of the longest Wikipedia articles ([https://en.wikipedia.org/w/index.php?title=Special:LongPages&limit=500&offset=0](https://en.wikipedia.org/w/index.php?title=Special:LongPages&limit=500&offset=0)).<br>
// MAGIC The most recent versions of the articles were extracted in XML format using Wikipedia's export tool: [https://en.wikipedia.org/wiki/Special:Export](https://en.wikipedia.org/wiki/Special:Export)
// MAGIC
// MAGIC Spark has support for importing XML files directly to DataFrames, [https://spark.apache.org/docs/latest/sql-data-sources-xml.html](https://spark.apache.org/docs/latest/sql-data-sources-xml.html).
// MAGIC
// MAGIC #### Definition for a word to be considered in this task
// MAGIC
// MAGIC A word is to be considered (and included in the counts) in this task if<br>
// MAGIC
// MAGIC - when the following punctuation characters are removed: '`.`', '`,`', '`;`', '`:`', '`!`', '`?`', '`(`', '`)`', '`[`', '`]`', '`{`', '`}`',<br>
// MAGIC - and all letters have been changed to lower case, i.e., `A` -> `a`, ...
// MAGIC
// MAGIC the word fulfils the following conditions:
// MAGIC
// MAGIC - the word contains only letters in the English alphabet: '`a`', ..., '`z`'
// MAGIC - the word is at least 5 letters long
// MAGIC - the word is not the English word for a specific month:<br>
// MAGIC     `january`, `february`, `march`, `april`, `may`, `june`, `july`, `august`, `september`, `october`, `november`, `december`
// MAGIC - the word in not the English word for a specific season: `summer`, `autumn`, `winter`, `spring`
// MAGIC
// MAGIC For example, words `(These` and `country,` would be valid words to consider with these rules (as `these` and `country`).
// MAGIC
// MAGIC In this task, you can assume that each line in an article is separated by the new line character, '`\n`'.<br>
// MAGIC And that each word is separated by a whitespace character, '` `'.
// MAGIC
// MAGIC #### The tasks
// MAGIC
// MAGIC Load the content of the Wikipedia articles, and find the answers to the following questions using the presented criteria for a word:
// MAGIC
// MAGIC - What are the 10 most frequent words across all included articles?
// MAGIC     - Give the answer as a data frame with columns `word` and `total_count`.
// MAGIC <the other questions were done by other teammates and their solutions won't be shown here>
// MAGIC
// MAGIC Even though the tasks ask for data frame answers, RDDs or Datasets can be helpful. However, their use is optional, and all the tasks can be completed by only using data frames.

// COMMAND ----------

val wikipediaDF: DataFrame = spark.read
  .option("rowTag", "page")
  .option("excludeAttribute", "true")
  .option("inferSchema", "true")
  .xml("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/wikipedia")
  .withColumn("date", to_date(col("revision.timestamp")))
  .withColumn("text", col("revision.text"))
  .select("id", "title", "date", "text")

// COMMAND ----------

// Transform texts into lowercase and remove special characters and English words mentioned above
val months: String = "january|february|march|april|may|june|july|august|september|october|november|december"
val seasons: String = "summer|autumn|winter|spring"

val processedWikipediaDF: DataFrame = wikipediaDF
  .withColumn("words", lower(col("text")))
  .withColumn("words", regexp_replace(col("words"), """[.,;:!?()\[\]{}]""", "")) // remove punctuation
  // .withColumn("words", regexp_replace(col("words"), s"\\b($months|$seasons)\\b", ""))
  .withColumn("words", regexp_replace(col("words"), "[ \n]+", " ")) // replace multiple consecutive whitespace with single whitespace
  .withColumn("words", split(col("words"), " "))
  .withColumn("words", filter(col("words"), w => w.rlike("^[a-z]{5,}$")))
  .withColumn("words", filter(col("words"), w => !w.rlike(s"($months|$seasons)")))

// COMMAND ----------

val tenMostFrequentWordsDF: DataFrame = processedWikipediaDF.select(explode(col("words")).as("word"))
  .groupBy("word")
  .agg(count("word").as("total_count"))
  .orderBy(col("total_count").desc)
  .limit(10)

// COMMAND ----------

println("Top 10 most frequent words across all articles:")
tenMostFrequentWordsDF.show(false)