package org.deeplearning4j.examples.misc.activationfunctions

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.util.{Collections, Random}

/**
  * This is an example that illustrates how to instantiate and use a custom activation function.
  * The example is identical to the one in org.deeplearning4j.examples.feedforward.regression.RegressionSum
  * except for the custom activation function
  */
object CustomActivationExample {
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

    val iterator = getTrainingData(batchSize, rng)

    //Create the network
    val numInput = 2
    val numOutputs = 1
    val nHidden = 10
    val net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.95)
      .list
      //INSTANTIATING CUSTOM ACTIVATION FUNCTION here as follows
      //Refer to CustomActivation class for more details on implementation
      .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
        .activation(new CustomActivation)
        .build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(nHidden).nOut(numOutputs)
        .build)
      .pretrain(false).backprop(true).build)

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
    val inputNDArray1: INDArray = Nd4j.create(input1, Array[Int](nSamples, 1))
    val inputNDArray2: INDArray = Nd4j.create(input2, Array[Int](nSamples, 1))
    val inputNDArray: INDArray = Nd4j.hstack(inputNDArray1, inputNDArray2)
    val outPut: INDArray = Nd4j.create(sum, Array[Int](nSamples, 1))
    val dataSet: DataSet = new DataSet(inputNDArray, outPut)
    val listDs = dataSet.asList
    Collections.shuffle(listDs, rng)
    new ListDataSetIterator(listDs, batchSize)
  }
}
