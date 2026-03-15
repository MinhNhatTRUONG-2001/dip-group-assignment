// Databricks notebook source
// MAGIC %md
// MAGIC Copyright 2025 Tampere University<br>
// MAGIC This notebook and software was developed for a Tampere University course COMP.CS.320.<br>
// MAGIC This source code is licensed under the MIT license. See LICENSE in the exercise repository root directory.<br>
// MAGIC Author(s): Ville Heikkilä \([ville.heikkila@tuni.fi](mailto:ville.heikkila@tuni.fi))

// COMMAND ----------

// add other required imports here
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, DoubleType, DateType, StructField, StructType}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 1 - Video game sales data
// MAGIC
// MAGIC The CSV file `assignment/sales/video_game_sales_2024.csv` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains video game sales data.<br>
// MAGIC The data is based on [https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024) dataset which is made available under the ODC Attribution License, [https://opendatacommons.org/licenses/by/1-0/index.html](https://opendatacommons.org/licenses/by/1-0/index.html). The data used in this task includes only the video games for which at least some sales data is available, and some original columns have been removed.
// MAGIC
// MAGIC Load the data from the CSV file into a data frame. The column headers and the first few data lines should give sufficient information about the source dataset. The numbers in the sales columns are given in millions.
// MAGIC
// MAGIC Using the data, find answers to the following:
// MAGIC
// MAGIC - Which publisher has the highest total sales in video games in Japan, considering games released in years 2001-2010?
// MAGIC - Separating games released in different years and considering only this publisher and only games released in years 2001-2010, what are the total sales, in Japan and globally, for each year? And how much of those global sales were for PlayStation 2 (PS2) games?
// MAGIC     - I.e., what are the total sales in Japan, in total globally, and in total for PS2 games, for video games released by this publisher in year 2001?<br>
// MAGIC       And the same for year 2002? ...
// MAGIC     - If some sales value is empty (i.e., NULL), it can be considered as 0 sales for that game in that region.

// COMMAND ----------

val videoGameSalesSchema = new StructType(Array(
  new StructField("title", StringType),
  new StructField("console", StringType),
  new StructField("genre", StringType),
  new StructField("publisher", StringType),
  new StructField("developer", StringType),
  new StructField("na_sales", DoubleType),
  new StructField("jp_sales", DoubleType),
  new StructField("pal_sales", DoubleType),
  new StructField("other_sales", DoubleType),
  new StructField("release_date", DateType)
))

val videoGameSalesDF: DataFrame = spark.read
  .option("header", "true")
  .option("sep", "|")
  .schema(videoGameSalesSchema)
  .csv("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/sales/video_game_sales_2024.csv")
  .select("console", "publisher", "na_sales", "jp_sales", "pal_sales", "other_sales", "release_date")

val videoGameSalesFrom2001To2010DF: DataFrame = videoGameSalesDF.filter(year(col("release_date")) >= 2001 && year(col("release_date")) <= 2010).na.fill(0)

// COMMAND ----------

val bestJapanPublisher: String = videoGameSalesFrom2001To2010DF.groupBy("publisher")
  .agg(sum(col("jp_sales")).as("total_sales_in_japan"))
  .agg(max_by(col("publisher"), col("total_sales_in_japan")).as("max_sales"))
  .first
  .getString(0)

val bestJapanPublisherSales: DataFrame = videoGameSalesFrom2001To2010DF.filter(col("publisher") === bestJapanPublisher) 
  .groupBy(year(col("release_date")).as("year"))
  .agg(
    round(sum("jp_sales"), 2).as("japan_total"),
    round(sum(col("na_sales") + col("jp_sales") + col("pal_sales") + col("other_sales")), 2).as("global_total"),
    round(sum(when(col("console") === "PS2", col("na_sales") + col("jp_sales") + col("pal_sales") + col("other_sales"))), 2).as("ps2_total")
  )
  .orderBy("year")

// COMMAND ----------

println(s"The publisher with the highest total video game sales in Japan is: '${bestJapanPublisher}'")
println("Sales data for this publisher:")
bestJapanPublisherSales.show()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Basic Task 4 - Football data and the best goalscorers in Spain and Italy
// MAGIC
// MAGIC The folder `assignment/football/events` in the [Shared container](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fe0c78478-e7f8-429c-a25f-015eae9f54bb%2FresourceGroups%2Ftuni-cs320-f2025-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Ftunics320f2025gen2/path/shared/etag/%220x8DE01A3A1A66C90%22/defaultId//publicAccessVal/None) contains information about events in [football](https://en.wikipedia.org/wiki/Association_football) matches during the season 2017-18 in five European top-level leagues: English Premier League, Italian Serie A, Spanish La Liga, German Bundesliga, and French Ligue 1. The data is based on a dataset from [https://figshare.com/collections/Soccer_match_event_dataset/4415000/5](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5). The data is given in Parquet format.
// MAGIC
// MAGIC Additional player related information are given in Parquet format at folder `assignment/football/players`. This dataset contains information about the player names, default roles when playing, and their birth areas.
// MAGIC
// MAGIC #### Background information
// MAGIC
// MAGIC In the considered leagues, a season is played in a double round-robin format where each team plays against all other teams twice. Once as a home team in their own stadium, and once as an away team in the other team's stadium. A season usually starts in August and ends in May.
// MAGIC
// MAGIC Each league match consists of two halves of 45 minutes each. Each half runs continuously, meaning that the clock is not stopped when the ball is out of play. The referee of the match may add some additional time to each half based on game stoppages. \[[https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time](https://en.wikipedia.org/wiki/Association_football#90-minute_ordinary_time)\]
// MAGIC
// MAGIC The team that scores more goals than their opponent wins the match.
// MAGIC
// MAGIC **Columns in the event data**
// MAGIC
// MAGIC Each row in the given data represents an event in a specific match. An event can be, for example, a pass, a foul, a shot, or a save attempt.<br>
// MAGIC Simple explanations for the available columns. Not all of these will be needed in this assignment.
// MAGIC
// MAGIC | column name | column type | description |
// MAGIC | ----------- | ----------- | ----------- |
// MAGIC | competition | string | The name of the competition |
// MAGIC | season | string | The season the match was played |
// MAGIC | matchId | integer | A unique id for the match |
// MAGIC | eventId | integer | A unique id for the event |
// MAGIC | homeTeam | string | The name of the home team |
// MAGIC | awayTeam | string | The name of the away team |
// MAGIC | event | string | The main category for the event |
// MAGIC | subEvent | string | The subcategory for the event |
// MAGIC | eventTeam | string | The name of the team that initiated the event |
// MAGIC | eventPlayerId | integer | The id for the player who initiated the event, 0 for events not identified to a single player |
// MAGIC | eventPeriod | string | `1H` for events in the first half, `2H` for events in the second half |
// MAGIC | eventTime | double | The event time in seconds counted from the start of the half |
// MAGIC | tags | array of strings | The descriptions of the tags associated with the event |
// MAGIC | startPosition | struct | The event start position given in `x` and `y` coordinates in range \[0,100\] |
// MAGIC | enPosition | struct | The event end position given in `x` and `y` coordinates in range \[0,100\] |
// MAGIC
// MAGIC The used event categories can be seen from `assignment/football/metadata/eventid2name.csv`.<br>
// MAGIC And all available tag descriptions from `assignment/football/metadata/tags2name.csv`.<br>
// MAGIC You don't need to access these files in the assignment, but they can provide context for the following basic tasks that will use the event data.
// MAGIC
// MAGIC Note that there are two events related to each goal that happened in the matches covered by the dataset.
// MAGIC
// MAGIC - One event for the player who scored the goal. This includes possible own goals, i.e., accidentally directing the ball to their own goal.
// MAGIC - One event for the goalkeeper who tried to stop the goal.
// MAGIC
// MAGIC **Columns in the player data**
// MAGIC
// MAGIC Each row represents a single player. All the columns will not be needed in the assignment.
// MAGIC
// MAGIC | column name  | column type | description |
// MAGIC | ------------ | ----------- | ----------- |
// MAGIC | playerId     | integer     | A unique id for the player |
// MAGIC | firstName    | string      | The first name of the player |
// MAGIC | lastName     | string      | The last name of the player |
// MAGIC | birthArea    | string      | The birth area (nation or similar) of the player |
// MAGIC | role         | string      | The main role of the player, either `Goalkeeper`, `Defender`, `Midfielder`, or `Forward` |
// MAGIC | foot         | string      | The stronger foot of the player |
// MAGIC
// MAGIC #### The task
// MAGIC
// MAGIC Using the given football data
// MAGIC
// MAGIC - Find the 7 players who scored the highest number of goals in `Spanish La Liga` during season `2017-2018`.
// MAGIC - Find the 7 players who scored the highest number of goals in `Italian Serie A` during season `2017-2018`.
// MAGIC
// MAGIC Give the results as DataFrames, which have one row for each player and the following columns:
// MAGIC
// MAGIC | column name    | column type | description |
// MAGIC | -------------- | ----------- | ----------- |
// MAGIC | player         | string      | The name of the player (first name + last name) |
// MAGIC | team           | string      | The team that the player played for |
// MAGIC | goals          | integer     | The number of goals the player scored |
// MAGIC
// MAGIC In this task, you can assume that all the relevant players played for the same team for the entire season.

// COMMAND ----------

val eventDF: DataFrame = spark.read.parquet("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/football/events")
val playerDF: DataFrame = spark.read.parquet("abfss://shared@tunics320f2025gen2.dfs.core.windows.net/assignment/football/players")

// COMMAND ----------

val goalscorersDF: DataFrame = eventDF.as("e")
  .select("competition", "eventPlayerId", "eventTeam", "tags")
  .filter(array_contains(col("tags"), "Goal") && array_contains(col("tags"), "Accurate"))
  .groupBy("eventPlayerId", "eventTeam", "competition")
  .agg(count(col("tags")).as("goals"))
  .join(playerDF.select("playerId", "firstName", "lastName").as("p"), col("e.eventPlayerId") === col("p.playerId"))

val goalscorersSpainDF: DataFrame = goalscorersDF.filter(col("competition") === "Spanish La Liga")
  .select(concat(col("firstName"), lit(" "), col("lastName")).as("player"), col("eventTeam").as("club"), col("goals"))
  .sort(col("goals").desc)
  .limit(7)

val goalscorersItalyDF: DataFrame = goalscorersDF.filter(col("competition") === "Italian Serie A")
  .select(concat(col("firstName"), lit(" "), col("lastName")).as("player"), col("eventTeam").as("club"), col("goals"))
  .sort(col("goals").desc)
  .limit(7)

// COMMAND ----------

println("The top 7 goalscorers in Spanish La Liga in season 2017-18:")
goalscorersSpainDF.show(false)

// COMMAND ----------

println("The top 7 goalscorers in Italian Serie A in season 2017-18:")
goalscorersItalyDF.show(false)