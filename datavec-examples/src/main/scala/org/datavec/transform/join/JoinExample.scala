package org.datavec.transform.join

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.join.Join
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.joda.time.DateTimeZone

import scala.collection.JavaConverters._

/**
  * This example shows how to perform joins in DataVec
  * Joins are analogous to join operations in databases/SQL: data from multiple sources are combined together, based
  * on some common key that appears in both sources.
  *
  * This example loads data from two CSV files. It is some mock customer data
  *
  * @author Alex Black
  */
object JoinExample {

  @throws[Exception]
  def main(args: Array[String]): Unit = {

    val customerInfoPath = new ClassPathResource("JoinExample/CustomerInfo.csv").getFile.getPath
    val purchaseInfoPath = new ClassPathResource("JoinExample/CustomerPurchases.csv").getFile.getPath

    //First: Let's define our two data sets, and their schemas

    val customerInfoSchema = new Schema.Builder()
      .addColumnLong("customerID")
      .addColumnString("customerName")
      .addColumnCategorical("customerCountry", List("USA", "France", "Japan", "UK").asJava)
      .build

    val customerPurchasesSchema = new Schema.Builder()
      .addColumnLong("customerID")
      .addColumnTime("purchaseTimestamp", DateTimeZone.UTC)
      .addColumnLong("productID")
      .addColumnInteger("purchaseQty")
      .addColumnDouble("unitPriceUSD")
      .build

    //Spark Setup
    val conf = new SparkConf
    conf.setMaster("local[*]")
    conf.setAppName("DataVec Join Example")
    val sc = new JavaSparkContext(conf)

    //Load the data:
    val rr = new CSVRecordReader
    val customerInfo = sc.textFile(customerInfoPath).map(new StringToWritablesFunction(rr))
    val purchaseInfo = sc.textFile(purchaseInfoPath).map(new StringToWritablesFunction(rr))
         //Collect data for later printing
    val customerInfoList = customerInfo.collect
    val purchaseInfoList = purchaseInfo.collect

    //Let's join these two data sets together, by customer ID
    val join: Join = new Join.Builder(Join.JoinType.Inner)
      .setJoinColumns("customerID")
      .setSchemas(customerInfoSchema, customerPurchasesSchema)
      .build

    val joinedData = SparkTransformExecutor.executeJoin(join, customerInfo, purchaseInfo)
    val joinedDataList = joinedData.collect

    //Stop spark, and wait a second for it to stop logging to console
    sc.stop()
    Thread.sleep(2000)


    //Print the original data
    println("\n\n----- Customer Information -----")
    println("Source file: " + customerInfoPath)
    println(customerInfoSchema)
    println("Customer Information Data:")
    for (line <- customerInfoList.asScala) {
      println(line)
    }


    println("\n\n----- Purchase Information -----")
    println("Source file: " + purchaseInfoPath)
    println(customerPurchasesSchema)
    println("Purchase Information Data:")
    for (line <- purchaseInfoList.asScala) {
      println(line)
    }


    //Print the joined data
    println("\n\n----- Joined Data -----")
    println(join.getOutputSchema)
    println("Joined Data:")
    for (line <- joinedDataList.asScala) {
      println(line)
    }
  }
}
