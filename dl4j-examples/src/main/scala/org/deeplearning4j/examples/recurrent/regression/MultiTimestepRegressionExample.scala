package org.deeplearning4j.examples.recurrent.regression

import org.apache.commons.io.FileUtils
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.jfree.ui.RefineryUtilities
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util
import javax.swing._

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * This example was inspired by Jason Brownlee's regression examples for Keras, found here:
  * http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
  * <p>
  * It demonstrates multi time step regression using LSTM
  */
object MultiTimestepRegressionExample {
  private val LOGGER = LoggerFactory.getLogger(MultiTimestepRegressionExample.getClass)
  private val baseDir = new File("dl4j-examples/src/main/resources/rnnRegression")
  private val baseTrainDir = new File(baseDir, "multiTimestepTrain")
  private val featuresDirTrain = new File(baseTrainDir, "features")
  private val labelsDirTrain = new File(baseTrainDir, "labels")
  private val baseTestDir = new File(baseDir, "multiTimestepTest")
  private val featuresDirTest = new File(baseTestDir, "features")
  private val labelsDirTest = new File(baseTestDir, "labels")

  private var numOfVariables = 0 // in csv.

  @throws[Exception]
  def main(args: Array[String]) {

    //Set number of examples for training, testing, and time steps
    val trainSize = 100
    val testSize = 20
    val numberOfTimesteps = 20

    //Prepare multi time step data, see method comments for more info
    val rawStrings = prepareTrainAndTest(trainSize, testSize, numberOfTimesteps)

    //Make sure miniBatchSize is divisable by trainSize and testSize,
    //as rnnTimeStep will not accept different sized examples
    val miniBatchSize: Int = 10
    // ----- Load the training data -----
    val trainFeatures = new CSVSequenceRecordReader
    trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath + "/train_%d.csv", 0, trainSize - 1))
    val trainLabels = new CSVSequenceRecordReader
    trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath + "/train_%d.csv", 0, trainSize - 1))
    val trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

    //Normalize the training data
    val normalizer = new NormalizerMinMaxScaler(0, 1)
    normalizer.fitLabel(true)
    normalizer.fit(trainDataIter) //Collect training data statistics
    trainDataIter.reset()


    // ----- Load the test data -----
    //Same process as for the training data.
    val testFeatures = new CSVSequenceRecordReader
    testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath + "/test_%d.csv", trainSize, trainSize + testSize - 1))
    val testLabels = new CSVSequenceRecordReader
    testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath + "/test_%d.csv", trainSize, trainSize + testSize - 1))

    val testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)

    trainDataIter.setPreProcessor(normalizer)
    testDataIter.setPreProcessor(normalizer)


    // ----- Configure the network -----
    val conf = new NeuralNetConfiguration.Builder()
      .seed(140)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .learningRate(0.15)
      .list
      .layer(0, new GravesLSTM.Builder()
        .activation(Activation.TANH).nIn(numOfVariables).nOut(10)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(10).nOut(numOfVariables)
        .build()).build()

    val net = new MultiLayerNetwork(conf)
    net.init()

    net.setListeners(new ScoreIterationListener(20))

    // ----- Train the network, evaluating the test set performance at each epoch -----
    val nEpochs = 50
    for (i <- 0 until nEpochs) {
      net.fit(trainDataIter)
      trainDataIter.reset()
      LOGGER.info("Epoch " + i + " complete. Time series evaluation:")

      //Run regression evaluation on our single column input
      val evaluation = new RegressionEvaluation(1)

      //Run evaluation. This is on 25k reviews, so can take some time
      while (testDataIter.hasNext) {
        val t = testDataIter.next
        val features = t.getFeatureMatrix
        val lables = t.getLabels
        val predicted = net.output(features, true)

        evaluation.evalTimeSeries(lables, predicted)
      }

      println(evaluation.stats)

      testDataIter.reset()
    }
    /**
      * All code below this point is only necessary for plotting
      */
    //Init rrnTimeStemp with train data and predict test data
    while (trainDataIter.hasNext) {
      val t = trainDataIter.next
      net.rnnTimeStep(t.getFeatureMatrix)
    }

    trainDataIter.reset()

    val t = testDataIter.next
    val predicted = net.rnnTimeStep(t.getFeatureMatrix)
    normalizer.revertLabels(predicted)

    //Convert raw string data to IndArrays for plotting
    val trainArray = createIndArrayFromStringList(rawStrings, 0, trainSize)
    val testArray = createIndArrayFromStringList(rawStrings, trainSize, testSize)

    //Create plot with out data
    val c = new XYSeriesCollection
    createSeries(c, trainArray, 0, "Train data")
    createSeries(c, testArray, trainSize - 1, "Actual test data")
    createSeries(c, predicted, trainSize - 1, "Predicted test data")

    plotDataset(c)

    LOGGER.info("----- Example Complete -----")
  }

  /**
    * Creates an IndArray from a list of strings
    * Used for plotting purposes
    */
  private def createIndArrayFromStringList(rawStrings: util.List[String], startIndex: Int, length: Int): INDArray = {
    val stringList = rawStrings.subList(startIndex, startIndex + length)
    val primitives = mutable.ArrayBuffer.fill[mutable.ArrayBuffer[Double]](numOfVariables)(
                      mutable.ArrayBuffer.fill[Double](stringList.size)(0.0))
    for (i <- stringList.asScala.indices) {
      val vals: Array[String] = stringList.get(i).split(",")
      for (j <- vals.indices) {
        primitives(j)(i) = java.lang.Double.valueOf(vals(j))
      }
    }
    Nd4j.create(Array[Int](1, length), primitives.map(_.toArray).toArray:_*)
  }

  /**
    * Used to create the different time series for ploting purposes
    */
  private def createSeries(seriesCollection: XYSeriesCollection, data: INDArray, offset: Int, name: String): XYSeriesCollection = {
    val nRows: Int = data.shape()(2)
    val predicted: Boolean = name.startsWith("Predicted")
    val repeat: Int = if (predicted) data.shape()(1)
    else data.shape()(0)
    for (j <- 0 until repeat) {
      val series = new XYSeries(name + j)
      for (i <- 0 until nRows) {
        if (predicted)
          series.add(i + offset, data.slice(0).slice(j).getDouble(i))
        else
          series.add(i + offset, data.slice(j).getDouble(i))
      }
      seriesCollection.addSeries(series)
    }
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
    val plot = chart.getXYPlot
    // Auto zoom to fit time series in initial window
    val rangeAxis = plot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setAutoRange(true)
    val panel = new ChartPanel(chart)
    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Training Data")
    RefineryUtilities.centerFrameOnScreen(f)
    f.setVisible(true)
  }

  /**
    * This method shows how you based on a CSV file can preprocess your data the structure expected for a
    * multi time step problem. This examples uses a single column CSV as input, but the example should be easy to modify
    * for use with a multi column input as well.
    *
    * @return
    * @throws IOException
    */
  @throws[IOException]
  private def prepareTrainAndTest(trainSize: Int, testSize: Int, numberOfTimesteps: Int): util.List[String] = {
    val rawPath = Paths.get(baseDir.getAbsolutePath + "/passengers_raw.csv")

    val rawStrings = Files.readAllLines(rawPath, Charset.defaultCharset)
    setNumOfVariables(rawStrings)

    //Remove all files before generating new ones
    FileUtils.cleanDirectory(featuresDirTrain)
    FileUtils.cleanDirectory(labelsDirTrain)
    FileUtils.cleanDirectory(featuresDirTest)
    FileUtils.cleanDirectory(labelsDirTest)

    for (i <- 0 until trainSize) {
      val featuresPath = Paths.get(featuresDirTrain.getAbsolutePath + "/train_" + i + ".csv")
      val labelsPath = Paths.get(labelsDirTrain + "/train_" + i + ".csv")
      for (j <- 0 until numberOfTimesteps) {
        Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator).getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
      }
      Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator).getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
    }

    for (i <- trainSize until testSize + trainSize) {
      val featuresPath = Paths.get(featuresDirTest + "/test_" + i + ".csv")
      val labelsPath = Paths.get(labelsDirTest + "/test_" + i + ".csv")
      for (j <- 0 until numberOfTimesteps) {
        Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator).getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
      }
      Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator).getBytes, StandardOpenOption.APPEND, StandardOpenOption.CREATE)
    }
    return rawStrings
  }

  private def setNumOfVariables(rawStrings: util.List[String]) {
    numOfVariables = rawStrings.get(0).split(",").length
  }
}
