package org.datavec.transform.debugging

import java.util

import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.{CategoricalColumnCondition, DoubleColumnCondition}
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform
import org.datavec.api.writable.DoubleWritable
import org.joda.time.{DateTimeFieldType, DateTimeZone}

import scala.collection.JavaConverters._

/**
  * This is a simple example for the DataVec transformation functionality (building on BasicDataVecExample)
  * It is designed to simply demonstrate that it is possible to obtain the schema after each step of a transform process.
  * This can be useful for debugging your TransformProcess scripts.
  *
  * @author Alex Black
  */
object PrintSchemasAtEachStep {
  def main(args: Array[String]) {
    //Define the Schema and TransformProcess as per BasicDataVecExample
    val inputDataSchema = new Schema.Builder()
      .addColumnsString("DateTimeString", "CustomerID", "MerchantID")
      .addColumnInteger("NumItemsInTransaction")
      .addColumnCategorical("MerchantCountryCode", List("USA", "CAN", "FR", "MX").asJava)
      .addColumnDouble("TransactionAmountUSD", 0.0, null, false, false) //$0.0 or more, no maximum limit, no NaN and no Infinite values
      .addColumnCategorical("FraudLabel", List("Fraud", "Legit").asJava)
      .build

    val tp = new TransformProcess.Builder(inputDataSchema)
      .removeColumns("CustomerID", "MerchantID")
      .filter(new ConditionFilter(new CategoricalColumnCondition("MerchantCountryCode", ConditionOp.NotInSet, Set("USA", "CAN").asJava)))
      .conditionalReplaceValueTransform(
        "TransactionAmountUSD", //Column to operate on
        new DoubleWritable(0.0), //New value to use, when the condition is satisfied
        new DoubleColumnCondition("TransactionAmountUSD", ConditionOp.LessThan, 0.0)) //Condition: amount < 0.0
      .stringToTimeTransform("DateTimeString", "YYYY-MM-DD HH:mm:ss.SSS", DateTimeZone.UTC)
      .renameColumn("DateTimeString", "DateTime")
      .transform(new DeriveColumnsFromTimeTransform.Builder("DateTime").addIntegerDerivedColumn("HourOfDay", DateTimeFieldType.hourOfDay).build)
      .removeColumns("DateTime")
      .build


    //Now, print the schema after each time step:
    val numActions = tp.getActionList.size
    (0 until numActions).foreach { i =>
      println("\n\n==================================================")
      println("-- Schema after step " + i + " (" + tp.getActionList.get(i) + ") --")

      println(tp.getSchemaAfterStep(i))
    }


    println("DONE.")
  }

}
