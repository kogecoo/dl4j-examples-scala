package org.nd4j.examples

import java.util

import org.nd4j.linalg.api.ops.impl.transforms.Sin
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
  * --- Nd4j Example 4: Additional Operations with INDArrays ---
  *
  * In this example, we'll see ways to manipulate INDArray
  *
  * @author Alex Black
  */
object Nd4jEx4_Ops {

  def main(args: Array[String]): Unit = {
    /*
    ND4J defines a wide variety of operations. Here we'll see how to use some of them:
    - Elementwise operations:   add, multiply, divide, subtract, etc
    add, mul, div, sub,
    INDArray.add(INDArray), INDArray.mul(INDArray), etc
    - Matrix multiplication:    mmul
    - Row/column vector ops:    addRowVector, mulColumnVector, etc
    - Element-wise transforms, like tanh, scalar max operations, etc
     */

    //First, let's see how in-place vs. copy operations work
    //Consider the calls:   myArray.add(1.0)    vs  myArray.addi(1.0)
    // i.e., "add" vs. "addi"   ->  the "i" means in-place.
    //In practice: the in-place ops modify the original array; the others ("copy ops") make a copy
    var originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5) //As per example 3
    val copyAdd = originalArray.add(1.0)
    println("Same object returned by add:    " + (originalArray eq copyAdd))
    println("Original array after originalArray.add(1.0):\n" + originalArray)
    println("copyAdd array:\n" + copyAdd)


        //Let's do the same thing with the in-place add operation:
    val inPlaceAdd = originalArray.addi(1.0)
    println()
    println("Same object returned by addi:    " + (originalArray eq inPlaceAdd)) //addi returns the exact same Java object
    println("Original array after originalArray.addi(1.0):\n" + originalArray)
    println("inPlaceAdd array:\n" + copyAdd)


    //Let's recreate our our original array for the next section, and create another one:
    originalArray = Nd4j.linspace(1, 15, 15).reshape('c', 3, 5)
    val random = Nd4j.rand(3, 5) //See example 2; we have a 3x5 with uniform random (0 to 1) values



    //We can perform element-wise operations. Note that the array shapes must match here
    // add vs. addi works in exactly the same way as for scalars
    val added = originalArray.add(random)
    println("\n\n\nRandom values:\n" + random)
    println("Original plus random values:\n" + added)


    //Matrix multiplication is easy:
    val first = Nd4j.rand(3, 4)
    val second = Nd4j.rand(4, 5)
    val mmul = first.mmul(second)
    println("\n\n\nShape of mmul array:      " + util.Arrays.toString(mmul.shape)) //3x5 output as expected


    //We can do row-wise ("for each row") and column-wise ("for each column") operations
    //Again, inplace vs. copy ops work the same way (i.e., addRowVector vs. addiRowVector)
    val row = Nd4j.linspace(0, 4, 5)
    println("\n\n\nRow:\n" + row)
    val mulRowVector = originalArray.mulRowVector(row) //For each row in 'originalArray', do an element-wise multiplication with the row vector
    println("Result of originalArray.mulRowVector(row)")
    println(mulRowVector)


    //Element-wise transforms are things like 'tanh' and scalar max values. These can be applied in a few ways:
    println("\n\n\n")
    println("Random array:\n" + random) //Again, note the limited printing precision, as per example 2
    println("Element-wise tanh on random array:\n" + Transforms.tanh(random))
    println("Element-wise power (x^3.0) on random array:\n" + Transforms.pow(random, 3.0))
    println("Element-wise scalar max (with scalar 0.5):\n" + Transforms.max(random, 0.5))
         //We can perform this in a more verbose way, too:
    val sinx = Nd4j.getExecutioner.execAndReturn(new Sin(random.dup))
    println("Element-wise sin(x) operation:\n" + sinx)
  }
}
