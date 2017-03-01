package org.nd4j.examples

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions

/**
  * --- Nd4j Example 6: Boolean Indexing ---
  *
  * In this example, we'll see ways to use boolean indexing to perform some simple conditional element-wise operations
  *
  * @author Alex Black
  */
object Nd4jEx6_BooleanIndexing {

  def main(args: Array[String]): Unit = {

    val nRows: Int = 3
    val nCols: Int = 5
    val rngSeed: Long = 12345

    //Generate random numbers between -1 and +1
    val random: INDArray = Nd4j.rand(nRows, nCols, rngSeed).muli(2).subi(1)

    println("Array values:")
    println(random)

    //For example, we can conditionally replace values less than 0.0 with 0.0:
    var randomCopy: INDArray = random.dup
    BooleanIndexing.replaceWhere(randomCopy, 0.0, Conditions.lessThan(0.0))
    println("After conditionally replacing negative values:\n" + randomCopy)

    //Or conditionally replace NaN values:
    val hasNaNs: INDArray = Nd4j.create(Array[Double](1.0, 1.0, Double.NaN, 1.0))
    BooleanIndexing.replaceWhere(hasNaNs, 0.0, Conditions.isNan)
    println("hasNaNs after replacing NaNs with 0.0:\n" + hasNaNs)

    //Or we can conditionally copy values from one array to another:
    randomCopy = random.dup
    val tens: INDArray = Nd4j.valueArrayOf(nRows, nCols, 10.0)
    BooleanIndexing.replaceWhere(randomCopy, tens, Conditions.lessThan(0.0))
    println("Conditionally copying values from array 'tens', if original value is less than 0.0\n" + randomCopy)


    //One simple task is to count the number of values that match the condition
    val op: MatchCondition = new MatchCondition(random, Conditions.greaterThan(0.0))
    val countGreaterThanZero: Int = Nd4j.getExecutioner.exec(op, Integer.MAX_VALUE).getInt(0) //MAX_VALUE = "along all dimensions" or equivalently "for entire array"
    println("Number of values matching condition 'greater than 0': " + countGreaterThanZero)
  }
}
