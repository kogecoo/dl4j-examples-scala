package org.deeplearning4j.examples.misc.lossfunctions

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.{Activation, IActivation}
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import java.util._

import scala.collection.mutable

/**
  * Created by susaneraly on 11/9/16.
  * This is an example that illustrates how to instantiate and use a custom loss function.
  * The example is identical to the one in org.deeplearning4j.examples.feedforward.regression.RegressionSum
  * except for the custom loss function
  */
object CustomLossExample {
  val seed: Int = 12345
  val iterations: Int = 1
  val nEpochs: Int = 500
  val nSamples: Int = 1000
  val batchSize: Int = 100
  val learningRate: Double = 0.001
  var MIN_RANGE: Int = 0
  var MAX_RANGE: Int = 3
  val rng: Random = new Random(seed)

  def main(args: Array[String]) {
    doTraining()
    //THE FOLLOWING IS TO ILLUSTRATE A SIMPLE GRADIENT CHECK.
    //It checks the implementation against the finite difference approximation, to ensure correctness
    //You will have to write your own gradient checks.
    //Use the code below and the following for reference.
    //  deeplearning4j/deeplearning4j-core/src/test/java/org/deeplearning4j/gradientcheck/LossFunctionGradientCheck.java
    doGradientCheck()
  }

  def doTraining() {

    val iterator: DataSetIterator = getTrainingData(batchSize, rng)

    //Create the network
    val numInput: Int = 2
    val numOutputs: Int = 1
    val nHidden: Int = 10
    val net: MultiLayerNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.95)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
        .activation(Activation.TANH)
        .build)
      //INSTANTIATE CUSTOM LOSS FUNCTION here as follows
      //Refer to CustomLossL1L2 class for more details on implementation
      .layer(1, new OutputLayer.Builder(new CustomLossL1L2)
      .activation(Activation.IDENTITY)
      .nIn(nHidden).nOut(numOutputs).build)
      .pretrain(false).backprop(true).build
    )
    net.init()
    net.setListeners(new ScoreIterationListener(100))


    //Train the network on the full data set, and evaluate in periodically
    for (i <- 0 until nEpochs) {
      iterator.reset()
      net.fit(iterator)
    }
    // Test the addition of 2 numbers (Try different numbers here)
    val input = Nd4j.create(Array[Double](0.111111, 0.3333333333333), Array[Int](1, 2))
    val out = net.output(input, false)
    println(out)
  }

  private def getTrainingData(batchSize: Int, rand: Random): DataSetIterator = {
    val sum = new Array[Double](nSamples)
    val input1 = new Array[Double](nSamples)
    val input2 = new Array[Double](nSamples)
    for (i <- 0 until nSamples) {
      input1(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble
      input2(i) = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble
      sum(i) = input1(i) + input2(i)
    }

    val inputNDArray1 = Nd4j.create(input1, Array[Int](nSamples, 1))
    val inputNDArray2 = Nd4j.create(input2, Array[Int](nSamples, 1))
    val inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2)
    val outPut = Nd4j.create(sum, Array[Int](nSamples, 1))
    val dataSet = new DataSet(inputNDArray, outPut)
    val listDs = dataSet.asList
    Collections.shuffle(listDs, rng)
    new ListDataSetIterator(listDs, batchSize)
  }

  def doGradientCheck() {
    val epsilon = 1e-3
    var totalNFailures = 0
    val maxRelError = 5.0 // in %
    val lossfn = new CustomLossL1L2
    val activationFns = Array[String]("identity", "softmax", "relu", "tanh", "sigmoid")
    val labelSizes = Array[Int](1, 2, 3, 4)
    for (i <- activationFns.indices) {
      println("Running checks for " + activationFns(i))
      val activation = Activation.fromString(activationFns(i)).getActivationFunction
      val labelList = makeLabels(activation, labelSizes)
      val preOutputList = makeLabels(new ActivationIdentity, labelSizes)
      for (j <- labelSizes.indices) {
        println("\tRunning check for length " + labelSizes(j))
        val label = labelList(j)
        val preOut = preOutputList(j)
        val grad = lossfn.computeGradient(label, preOut, activation, null)
        val iterPreOut = new NdIndexIterator(preOut.shape:_*)
        while (iterPreOut.hasNext) {
          val next = iterPreOut.next
          //checking gradient with total score wrt to each output feature in label
          val before = preOut.getDouble(next:_*)
          preOut.putScalar(next, before + epsilon)
          val scorePlus = lossfn.computeScore(label, preOut, activation, null, true)
          preOut.putScalar(next, before - epsilon)
          val scoreMinus = lossfn.computeScore(label, preOut, activation, null, true)
          preOut.putScalar(next, before)

          val scoreDelta = scorePlus - scoreMinus
          val numericalGradient = scoreDelta / (2 * epsilon)
          val analyticGradient = grad.getDouble(next:_*)
          var relError = Math.abs(analyticGradient - numericalGradient) * 100 / Math.abs(numericalGradient)
          if (analyticGradient == 0.0 && numericalGradient == 0.0) relError = 0.0
          if (relError > maxRelError || java.lang.Double.isNaN(relError)) {
            println("\t\tParam " + next.mkString + " FAILED: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient + ", relErrorPerc= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus)
            totalNFailures += 1
          } else {
            println("\t\tParam " + next.mkString + " passed: grad= " + analyticGradient + ", numericalGrad= " + numericalGradient + ", relError= " + relError + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus)
          }
        }
      }
    }
    if (totalNFailures > 0) println("DONE:\n\tGradient check failed for loss function; total num failures = " + totalNFailures)
    else println("DONE:\n\tSimple gradient check passed - This is NOT exhaustive by any means")
  }

  /* This function is a utility function for the gradient check above
     It generate labels randomly in the right range depending on the activation function
     Uses a gaussian
     identity: range is anything
     relu: range is non-negative
     softmax: range is non-negative and adds up to 1
     sigmoid: range is between 0 and 1
     tanh: range is between -1 and 1

   */
  def makeLabels(activation: IActivation, labelSize: Array[Int]): Array[INDArray] = {
    //edge cases are label size of one for everything except softmax which is two
    //+ve and -ve values, zero and non zero values, less than zero/greater than zero
    val returnVals = mutable.ArrayBuffer.empty[INDArray]
    for (i <- labelSize.indices) {
      val aLabelSize = labelSize(i)
      val r = new Random
      val someVals  = new Array[Double](aLabelSize)
      var someValsSum = 0.0
      for (j <- 0 until aLabelSize) {
        val someVal = r.nextGaussian
        val transformVal = activation.toString match {
          case "identity" => someVal
          case "softmax"  => someVal
          case "sigmoid"  => Math.sin(someVal)
          case "tanh"     => Math.tan(someVal)
          case "relu"     => someVal * someVal + 4
          case _          => throw new RuntimeException("Unknown activation function")
        }
        someVals(j) = transformVal
        someValsSum += someVals(j)
      }
      if ("sigmoid" == activation.toString) {
        for (j <- 0 until aLabelSize) {
          someVals(j) /= someValsSum
        }
      }
      returnVals += Nd4j.create(someVals)
    }
    returnVals.toArray
  }

}
