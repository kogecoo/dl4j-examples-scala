package org.nd4j.examples

import java.util.Arrays

import org.nd4j.linalg.api.ops.impl.indexaccum.IMin
import org.nd4j.linalg.factory.Nd4j

/**
  * --- Nd4j Example 5: Accumulation/Reduction Operations ---
  *
  * In this example, we'll see ways to reduce INDArrays - for example, perform sum and max operations
  *
  * @author Alex Black
  */
object Nd4jEx5_Accumulations {

  def main(args: Array[String]): Unit = {
    /*
    There are two types of accumulation/reduction operations:
    - Whole array operations                    ->  returns a scalar value
    - Operations along one or more dimensions   ->  returns an array

    Furthermore, there are two classes of accumulations:
    - Standard accumulations:   Accumulations that return a real-value - for example, min, max, sum, etc.
    - Index accumulations:      Accumulations that return an integer index - for example argmax

    */

    val originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5) //As per example 3
    println("Original array: \n" + originalArray)

    //First, let's consider whole array reductions:
    val minValue = originalArray.minNumber.doubleValue
    val maxValue = originalArray.maxNumber.doubleValue
    val sum = originalArray.sumNumber.doubleValue
    val avg = originalArray.meanNumber.doubleValue
    val stdev = originalArray.stdNumber.doubleValue

    println("minValue:       " + minValue)
    println("maxValue:       " + maxValue)
    println("sum:            " + sum)
    println("average:        " + avg)
    println("standard dev.:  " + stdev)


    //Second, let's perform the same along dimension 0.
    //In this case, the output is a [1,5] array; each output value is the min/max/mean etc of the corresponding column:
    val minAlong0 = originalArray.min(0)
    val maxAlong0 = originalArray.max(0)
    val sumAlong0 = originalArray.sum(0)
    val avgAlong0 = originalArray.mean(0)
    val stdevAlong0 = originalArray.std(0)

    println("\n\n\n")
    println("min along dimension 0:  " + minAlong0)
    println("max along dimension 0:  " + maxAlong0)
    println("sum along dimension 0:  " + sumAlong0)
    println("avg along dimension 0:  " + avgAlong0)
    println("stdev along dimension 0:  " + stdevAlong0)

    //If we had instead performed these along dimension 1, we would instead get a [3,1] array out
    //In this case, each output value would be the reduction of the values in each column
    //Again, note that when this is printed it looks like a row vector, but is in facta column vector
    val avgAlong1 = originalArray.mean(1)
    println("\n\navg along dimension 1:  " + avgAlong1)
    println("Shape of avg along d1:  " + Arrays.toString(avgAlong1.shape))



    //Index accumulations return an integer value.
    val argMaxAlongDim0 = Nd4j.argMax(originalArray, 0) //Index of the max value, along dimension 0
    println("\n\nargmax along dimension 0:   " + argMaxAlongDim0)
    val argMinAlongDim0 = Nd4j.getExecutioner.exec(new IMin(originalArray), 0) //Index of the min value, along dimension 0
    println("argmin along dimension 0:   " + argMinAlongDim0)
  }
}
