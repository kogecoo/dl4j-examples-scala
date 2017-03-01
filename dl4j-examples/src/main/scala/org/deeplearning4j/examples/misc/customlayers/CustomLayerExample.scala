package org.deeplearning4j.examples.misc.customlayers

import org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer
import org.deeplearning4j.gradientcheck.GradientCheckUtil
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.distribution.NormalDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.{File, IOException}
import java.util.Random

/**
  * Custom layer example. This example shows the use and some basic testing for a custom layer implementation.
  * For more details, see the CustomLayerExampleReadme.md file
  *
  * @author Alex Black
  */
object CustomLayerExample {

  //Double precision for the gradient checks. See comments in the doGradientCheck() method
  // See also http://nd4j.org/userguide.html#miscdatatype
  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  @throws[IOException]
  def main(args: Array[String]) {
    runInitialTests()
    doGradientCheck()
  }

  @throws[IOException]
  private def runInitialTests() {
    /*
    This method shows the configuration and use of the custom layer.
    It also shows some basic sanity checks and tests for the layer.
    In practice, these tests should be implemented as unit tests; for simplicity, we are just printing the results
     */

    println("----- Starting Initial Tests -----")

    val nIn = 5
    val nOut = 8

    //Let's create a network with our custom layer

    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .updater(Updater.RMSPROP)
      .rmsDecay(0.95)
      .weightInit(WeightInit.XAVIER)
      .regularization(true)
      .l2(0.03)
      .list
      .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build()) //Standard DenseLayer
      .layer(1, new CustomLayer.Builder().activation(Activation.TANH) //Property inherited from FeedForwardLayer
        .secondActivationFunction(Activation.SIGMOID) //Custom property we defined for our layer
        .nIn(6).nOut(7)
        .build() //nIn and nOut also inherited from FeedForwardLayer
      )
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //Standard OutputLayer
        .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
      .pretrain(false).backprop(true).build()

    //First:  run some basic sanity checks on the configuration:
    val customLayerL2 = config.getConf(1).getLayer.getL2
    println("l2 coefficient for custom layer: " + customLayerL2) //As expected: custom layer inherits the global L2 parameter configuration
    val customLayerUpdater: Updater = config.getConf(1).getLayer.getUpdater
    println("Updater for custom layer: " + customLayerUpdater) //As expected: custom layer inherits the global Updater configuration

    //Second: We need to ensure that that the JSON and YAML configuration works, with the custom layer
    // If there were problems with serialization, you'd get an exception during deserialization ("No suitable constructor found..." for example)
    val configAsJson: String = config.toJson()
    val configAsYaml: String = config.toYaml()
    val fromJson: MultiLayerConfiguration = MultiLayerConfiguration.fromJson(configAsJson)
    val fromYaml: MultiLayerConfiguration = MultiLayerConfiguration.fromYaml(configAsYaml)

    println("JSON configuration works: " + config.equals(fromJson))
    println("YAML configuration works: " + config.equals(fromYaml))

    val net: MultiLayerNetwork = new MultiLayerNetwork(config)
    net.init()

    //Third: Let's run some more basic tests. First, check that the forward and backward pass methods don't throw any exceptions
    // To do this: we'll create some simple test data
    val minibatchSize = 5
    val testFeatures = Nd4j.rand(minibatchSize, nIn)
    val testLabels = Nd4j.zeros(minibatchSize, nOut)
    val r = new Random(12345)
    for (i <- 0 until minibatchSize) {
      testLabels.putScalar(i, r.nextInt(nOut), 1) //Random one-hot labels data
    }

    val activations = net.feedForward(testFeatures)
    val activationsCustomLayer = activations.get(2) //Activations index 2: index 0 is input, index 1 is first layer, etc.
    println("\nActivations from custom layer:")
    println(activationsCustomLayer)
    net.fit(new DataSet(testFeatures, testLabels))

    //Finally, let's check the model serialization process, using ModelSerializer:
    ModelSerializer.writeModel(net, new File("CustomLayerModel.zip"), true)
    val restored: MultiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(new File("CustomLayerModel.zip"))

    println()
    println("Original and restored networks: configs are equal: " + net.getLayerWiseConfigurations == restored.getLayerWiseConfigurations)
    println("Original and restored networks: parameters are equal: " + net.params == restored.params)
  }

  private def doGradientCheck() {
    /*
    Gradient checks are one of the most important components of implementing a layer
    They are necessary to ensure that your implementation is correct: without them, you could easily have a subtle
     error, and not even know it.

    Deeplearning4j comes with a gradient check utility that you can use to check your layers.
    This utility works for feed-forward layers, CNNs, RNNs etc.
    For more details on gradient checks, and some references, see the Javadoc for the GradientCheckUtil class:
    https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java

    There are a few things to note when doing gradient checks:
    1. It is necessary to use double precision for ND4J. Single precision (float - the default) isn't sufficiently
       accurate for reliably performing gradient checks
    2. It is necessary to set the updater to None, or equivalently use both the SGD updater and learning rate of 1.0
       Reason: we are testing the raw gradients before they have been modified with learning rate, momentum, etc.
    */

    println("\n\n\n----- Starting Gradient Check -----")

    Nd4j.getRandom.setSeed(12345)
    val nIn = 3
    val nOut = 2

    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .updater(Updater.NONE)
      .learningRate(1.0)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0, 1))  //Larger weight init than normal can help with gradient checks
      .regularization(true)
      .l2(0.03)
      .list
      .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(3).build) //Standard DenseLayer
      .layer(1, new CustomLayer.Builder()
        .activation(Activation.TANH)                   //Property inherited from FeedForwardLayer
        .secondActivationFunction(Activation.SIGMOID)  //Custom property we defined for our layer
        .nIn(3).nOut(3)                                //nIn and nOut also inherited from FeedForwardLayer
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) //Standard OutputLayer
        .activation(Activation.SOFTMAX).nIn(3).nOut(nOut).build())
      .pretrain(false).backprop(true).build()

    val net = new MultiLayerNetwork(config)
    net.init()

    val print = true                         //Whether to print status for each parameter during testing
    val return_on_first_failure = false      //If true: terminate test on first failure
    val gradient_check_epsilon = 1e-8        //Epsilon value used for gradient checks
    val max_relative_error = 1e-5    //Maximum relative error allowable for each parameter
    val min_absolute_error = 1e-10   //Minimum absolute error, to avoid failures on 0 vs 1e-30, for example.

    //Create some random input data to use in the gradient check
    val minibatchSize = 5
    val features = Nd4j.rand(minibatchSize, nIn)
    val labels = Nd4j.zeros(minibatchSize, nOut)
    val r = new Random(12345)
    var i = 0
    for (i <- 0 until minibatchSize) {
      labels.putScalar(i, r.nextInt(nOut), 1)
    }

    //Print the number of parameters in each layer. This can help to identify the layer that any failing parameters
    // belong to.
    for (i <- 0 until 3) {
      println("# params, layer " + i + ":\t" + net.getLayer(i).numParams)
    }
    GradientCheckUtil.checkGradients(net, gradient_check_epsilon, max_relative_error, min_absolute_error, print,
      return_on_first_failure, features, labels)
  }

}
