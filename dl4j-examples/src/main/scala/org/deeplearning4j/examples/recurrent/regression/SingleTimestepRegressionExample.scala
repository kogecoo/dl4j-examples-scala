package org.deeplearning4j.examples.recurrent.regression

import org.datavec.api.records.reader.SequenceRecordReader
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.{PlotOrientation, XYPlot}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import java.io.File
import javax.swing._

/**
  * This example was inspired by Jason Brownlee's regression examples for Keras, found here:
  * http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
  *
  * It demonstrates single time step regression using LSTM
  */
object SingleTimestepRegressionExample {

  private val LOGGER = LoggerFactory.getLogger(SingleTimestepRegressionExample.getClass)
  private val baseDir = new File("dl4j-examples/src/main/resources/rnnRegression")

  @throws[Exception]
  def main(args: Array[String]) {

    val miniBatchSize = 32

    // ----- Load the training data -----
    val trainReader: SequenceRecordReader = new CSVSequenceRecordReader(0, ";")
    trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath + "/passengers_train_%d.csv", 0, 0))

    //For regression, numPossibleLabels is not used. Setting it to -1 here
    val trainIter: DataSetIterator = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true)

    val testReader: SequenceRecordReader = new CSVSequenceRecordReader(0, ";")
    testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath + "/passengers_test_%d.csv", 0, 0))
    val testIter: DataSetIterator = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true)

    //Create data set from iterator here since we only have a single data set
    val trainData = trainIter.next()
    val testData = testIter.next()

    //Normalize data, including labels (fitLabel=true)
    val normalizer = new NormalizerMinMaxScaler(0, 1)
    normalizer.fitLabel(true)
    normalizer.fit(trainData) //Collect training data statistics

    normalizer.transform(trainData)
    normalizer.transform(testData)

    // ----- Configure the network -----
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(140)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .learningRate(0.0015)
      .list
      .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(10).nOut(1)
        .build())
      .build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    net.setListeners(new ScoreIterationListener(20))

    // ----- Train the network, evaluating the test set performance at each epoch -----
    val nEpochs = 300
    for (i <- 0 until nEpochs) {
      net.fit(trainData)
      LOGGER.info("Epoch " + i + " complete. Time series evaluation:")

      //Run regression evaluation on our single column input
      val evaluation = new RegressionEvaluation(1)
      val features = testData.getFeatureMatrix

      val lables = testData.getLabels
      val predicted = net.output(features, false)

      evaluation.evalTimeSeries(lables, predicted)

      //Just do sout here since the logger will shift the shift the columns of the stats
      println(evaluation.stats)
    }

    //Init rrnTimeStemp with train data and predict test data
    net.rnnTimeStep(trainData.getFeatureMatrix)
    val predicted = net.rnnTimeStep(testData.getFeatureMatrix)

    //Revert data back to original values for plotting
    normalizer.revert(trainData)
    normalizer.revert(testData)
    normalizer.revertLabels(predicted)

    //Create plot with out data
    val c: XYSeriesCollection = new XYSeriesCollection
    createSeries(c, trainData.getFeatures, 0, "Train data")
    createSeries(c, testData.getFeatures, 99, "Actual test data")
    createSeries(c, predicted, 100, "Predicted test data")

    plotDataset(c)

    LOGGER.info("----- Example Complete -----")
  }

  private def createSeries(seriesCollection: XYSeriesCollection, data: INDArray, offset: Int, name: String): XYSeriesCollection = {
    val nRows = data.shape()(2)
    val series = new XYSeries(name)
    for (i <- 0 until nRows) {
      series.add(i + offset, data.getDouble(i))
    }
    seriesCollection.addSeries(series)
    seriesCollection
  }

  /**
    * Generate an xy plot of the datasets provided.
    */
  private def plotDataset(c: XYSeriesCollection) {

    val title = "Regression example"
    val xAxisLabel = "Timestep"
    val yAxisLabel = "Number of passengers"
    val orientation = PlotOrientation.VERTICAL
    val legend = true
    val tooltips = false
    val urls = false
    val chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls)

    // get a reference to the plot for further customisation...
    val plot: XYPlot = chart.getXYPlot

    // Auto zoom to fit time series in initial window
    val rangeAxis: NumberAxis = plot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setAutoRange(true)

    val panel = new ChartPanel(chart)

    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Training Data")

    RefineryUtilities.centerFrameOnScreen(f)
    f.setVisible(true) }
}
