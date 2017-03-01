package org.nd4j.examples

import java.util.Arrays

import org.nd4j.linalg.factory.Nd4j

/**
  * --- Nd4j Example 2: Creating INDArrays ---
  *
  * In this example, we'll see a number of different ways to create INDArrays
  *
  * @author Alex Black
  */
object Nd4jEx2_CreatingINDArrays {

  def main(args: Array[String]): Unit = {
    //Here, we'll see how to create INDArrays with different scalar value initializations
    val nRows = 3
    val nColumns = 5
    val allZeros = Nd4j.zeros(nRows, nColumns)
    println("Nd4j.zeros(nRows, nColumns)")
    println(allZeros)

    val allOnes = Nd4j.ones(nRows, nColumns)
    println("\nNd4j.ones(nRows, nColumns)")
    println(allOnes)

    val allTens = Nd4j.valueArrayOf(nRows, nColumns, 10.0)
    println("\nNd4j.valueArrayOf(nRows, nColumns, 10.0)")
    println(allTens)



    //We can also create INDArrays from double[] and double[][] (or, float/int etc Java arrays)
    val vectorDouble: Array[Double] = Array[Double](1, 2, 3)
    val rowVector = Nd4j.create(vectorDouble)
    println("rowVector:              " + rowVector)
    println("rowVector.shape():      " + Arrays.toString(rowVector.shape)) //1 row, 3 columns

    val columnVector = Nd4j.create(vectorDouble, Array[Int](3, 1)) //Manually specify: 3 rows, 1 column
    println("columnVector:           " + columnVector) //Note for printing: row/column vectors are printed as one line
    println("columnVector.shape():   " + Arrays.toString(columnVector.shape)) //3 row, 1 columns

    val matrixDouble: Array[Array[Double]] = Array[Array[Double]](Array(1.0, 2.0, 3.0), Array(4.0, 5.0, 6.0))
    val matrix = Nd4j.create(matrixDouble)
    println("\nINDArray defined from double[][]:")
    println(matrix)



    //It is also possible to create random INDArrays:
    //Be aware however that by default, random values are printed with truncated precision using INDArray.toString()
    val shape: Array[Int] = Array[Int](nRows, nColumns)
    val uniformRandom = Nd4j.rand(shape)
    println("\n\n\nUniform random array:")
    println(uniformRandom)
    println("Full precision of random value at position (0,0): " + uniformRandom.getDouble(0, 0))

    val gaussianMeanZeroUnitVariance = Nd4j.randn(shape)
    println("\nN(0,1) random array:")
    println(gaussianMeanZeroUnitVariance)

    //We can make things repeatable using RNG seed:
    val rngSeed: Long = 12345
    val uniformRandom2 = Nd4j.rand(shape, rngSeed)
    val uniformRandom3 = Nd4j.rand(shape, rngSeed)
    println("\nUniform random arrays with same fixed seed:")
    println(uniformRandom2)
    println()
    println(uniformRandom3)



    //Of course, we aren't restricted to 2d. 3d or higher is easy:
    val threeDimArray = Nd4j.ones(3, 4, 5)       //3x4x5 INDArray
    val fourDimArray = Nd4j.ones(3, 4, 5, 6)     //3x4x5x6 INDArray
    val fiveDimArray = Nd4j.ones(3, 4, 5, 6, 7)  //3x4x5x6x7 INDArray
    println("\n\n\nCreating INDArrays with more dimensions:")
    println("3d array shape:         " + Arrays.toString(threeDimArray.shape))
    println("4d array shape:         " + Arrays.toString(fourDimArray.shape))
    println("5d array shape:         " + Arrays.toString(fiveDimArray.shape))




    //We can create INDArrays by combining other INDArrays, too:
    val rowVector1 = Nd4j.create(Array[Double](1, 2, 3))
    val rowVector2 = Nd4j.create(Array[Double](4, 5, 6))

    val vStack = Nd4j.vstack(rowVector1, rowVector2) //Vertical stack:   [1,3]+[1,3] to [2,3]
    val hStack = Nd4j.hstack(rowVector1, rowVector2) //Horizontal stack: [1,3]+[1,3] to [1,6]
    println("\n\n\nCreating INDArrays from other INDArrays, using hstack and vstack:")
    println("vStack:\n" + vStack)
    println("hStack:\n" + hStack)

    //There's some other miscellaneous methods, too:
    val identityMatrix = Nd4j.eye(3)
    println("\n\n\nNd4j.eye(3):\n" + identityMatrix)
    val linspace = Nd4j.linspace(1, 10, 10) //Values 1 to 10, in 10 steps
    println("Nd4j.linspace(1,10,10):\n" + linspace)
    val diagMatrix = Nd4j.diag(rowVector2) //Create square matrix, with rowVector2 along the diagonal
    println("Nd4j.diag(rowVector2):\n" + diagMatrix)
  }

}
