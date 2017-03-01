package org.deeplearning4j.examples.feedforward.regression

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.examples.feedforward.regression.function.{MathFunction, SinXDivXMathFunction}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.util.{Collections, Random}
import javax.swing._

/** Example: Train a network to reproduce certain mathematical functions, and plot the results.
  * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
  * predictions as training progresses.
  * A number of mathematical functions are implemented here.
  * Note the use of the identity function on the network output layer, for regression
  *
  * @author Alex Black
  */
object RegressionMathFunctions {

  //Random number generator seed, for reproducability
  val seed = 12345
  //Number of iterations per minibatch
  val iterations = 1
  //Number of epochs (full passes of the data)
  val nEpochs = 2000
  //How frequently should we plot the network output?
  val plotFrequency = 500
  //Number of data points
  val nSamples = 1000
  //Batch size .e., each epoch has nSamples/batchSize parameter updates
  val batchSize = 100
  //Network learning rate
  val learningRate = 0.01
  val rng = new Random(seed)
  val numInputs = 1
  val numOutputs = 1

  def main(args: Array[String]) {

    //Switch these two options to do different functions with different networks
    val fn = new SinXDivXMathFunction
    val conf = getDeepDenseLayerNetworkConfiguration

    //Generate the training data
    val x = Nd4j.linspace(-10, 10, nSamples).reshape(nSamples, 1)
    val iterator = getTrainingData(x, fn, batchSize, rng)

    //Create the network
    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))


    //Train the network on the full data set, and evaluate in periodically
    val networkPredictions = new Array[INDArray](nEpochs / plotFrequency)
    for (i <- 0 until nEpochs) {
      iterator.reset()
      net.fit(iterator)
      if ((i + 1) % plotFrequency == 0) networkPredictions(i / plotFrequency) = net.output(x, false)
    }
    //Plot the target data and the network predictions
    plot(fn, x, fn.getFunctionValues(x), networkPredictions:_*)
  }

  /** Returns the network configuration, 2 hidden DenseLayers of size 50.
    */
  private def getDeepDenseLayerNetworkConfiguration: MultiLayerConfiguration = {
    val numHiddenNodes = 50
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation(Activation.TANH)
        .build)
      .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        .activation(Activation.TANH)
        .build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(numHiddenNodes).nOut(numOutputs)
        .build)
      .pretrain(false).backprop(true)
      .build
  }

  /** Create a DataSetIterator for training
    *
    * @param x         X values
    * @param function  Function to evaluate
    * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
    * @param rng       Random number generator (for repeatability)
    */
  private def getTrainingData(x: INDArray, function: MathFunction, batchSize: Int, rng: Random): DataSetIterator = {
    val y = function.getFunctionValues(x)
    val allData = new DataSet(x, y)
    val list = allData.asList
    Collections.shuffle(list, rng)
    new ListDataSetIterator(list, batchSize)
  }

  //Plot the data
  private def plot(function: MathFunction, x: INDArray, y: INDArray, predicted: INDArray*) {
    val dataSet = new XYSeriesCollection
    addSeries(dataSet, x, y, "True Function (Labels)")

    for (i <- predicted.indices) {
      addSeries(dataSet, x, predicted(i), String.valueOf(i))
    }

    val chart = ChartFactory.createXYLineChart(
      "Regression Example - " + function.getName, // chart title
      "X", // x axis label
      function.getName + "(X)", // y axis label
      dataSet, PlotOrientation.VERTICAL, // data
      true, // include legend
      true, // tooltips
      false // urls
    )

    val panel = new ChartPanel(chart)

    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()

    f.setVisible(true)
  }

  private def addSeries(dataSet: XYSeriesCollection, x: INDArray, y: INDArray, label: String) {
    val xd = x.data.asDouble
    val yd = y.data.asDouble
    val s = new XYSeries(label)
    var j= 0
    for (j <- xd.indices) {
      s.add(xd(j), yd(j))
    }
    dataSet.addSeries(s)
  }
}
