package org.nd4j.examples

import java.util

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

/**
  * --- Nd4j Example 3: Getting and setting parts of INDArrays ---
  *
  * In this example, we'll see ways to obtain and manipulate subsets of INDArray
  *
  * @author Alex Black
  */
object Nd4jEx3_GettingAndSettingSubsets {

  def main(args: Array[String]): Unit = {

    //Let's start by creating a 3x5 INDArray with manually specified values
    // To do this, we are starting with a 1x15 array, and perform a 'reshape' operation to convert it to a 3x5 INDArray
    var originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5)
    println("Original Array:")
    println(originalArray)

    //We can use getRow and getColumn operations to get a row or column respectively:
    val firstRow = originalArray.getRow(0)
    val lastColumn = originalArray.getColumn(4)
    println()
    println("First row:\n" + firstRow)
    println("Last column:\n" + lastColumn)
    //Careful of the printing here: lastColumn looks like a row vector when printed, but it's really a column vector
    println("Shapes:         " + util.Arrays.toString(firstRow.shape) + "\t" + util.Arrays.toString(lastColumn.shape))

    //A key concept in ND4J is the idea of views: one INDArray may point to the same locations in memory as other arrays
    //For example, getRow and getColumn are both views of originalArray
    //Consequently, changes to one results in changes to the other:
    firstRow.addi(1.0) //In-place addition operation: changes the values of both firstRow AND originalArray:
    println("\n\n")
    println("firstRow, after addi operation:")
    println(firstRow)
    println("originalArray, after firstRow.addi(1.0) operation: (note it is modified, as firstRow is a view of originalArray)")
    println(originalArray)



    //Let's recreate our our original array for the next section...
    originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5)


    //We can select arbitrary subsets, using INDArray indexing:
    //All rows, first 3 columns (note that internal here is columns 0 inclusive to 3 exclusive)
    val first3Columns = originalArray.get(NDArrayIndex.all, NDArrayIndex.interval(0, 3))
    println("first 3 columns:\n" + first3Columns)
    //Again, this is also a view:
    first3Columns.addi(100)
    println("originalArray, after first3Columns.addi(100)")
    println(originalArray)



    //Let's recreate our our original array for the next section...
    originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5)



    //We can similarly set arbitrary subsets.
    //Let's set the 3rd column (index 2) to zeros:
    val zerosColumn = Nd4j.zeros(3, 1)
    originalArray.put(Array[INDArrayIndex](NDArrayIndex.all, NDArrayIndex.point(2)), zerosColumn) //All rows, column index 2
    println("\n\n\nOriginal array, after put operation:\n" + originalArray)



    //Let's recreate our our original array for the next section...
    originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5)


    //Sometimes, we don't want this in-place behaviour. In this case: just add a .dup() operation at the end
    //the .dup() operation - aka 'duplicate' - creates a new and separate array
    val firstRowDup = originalArray.getRow(0).dup //We now have a copy of the first row. i.e., firstRowDup is NOT a view of originalArray
    firstRowDup.addi(100)
    println("\n\n\n")
    println("firstRowDup, after .addi(100):\n" + firstRowDup)
    println("originalArray, after firstRowDup.addi(100): (note it is unmodified)\n" + originalArray)
  }
}
