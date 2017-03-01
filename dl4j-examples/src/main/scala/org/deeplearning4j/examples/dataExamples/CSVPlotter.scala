package org.deeplearning4j.examples.dataExamples

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.{ChartFactory, ChartPanel}
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.{File, IOException}
import javax.swing._

import scala.collection.mutable

/**
  * Read a csv file. Fit and plot the data using Deeplearning4J.
  *
  * @author Robert Altena
  */
object CSVPlotter {

  @throws[IOException]
  @throws[InterruptedException]
  def main(args: Array[String]) {
    val filename: String = new ClassPathResource("/DataExamples/CSVPlotData.csv").getFile.getPath
    val ds: DataSet = readCSVDataset(filename)

    val DataSetList = mutable.ArrayBuffer.empty[DataSet]
    DataSetList += ds

    plotDataset(DataSetList.toArray) //Plot the data, make sure we have the right data.

    val net: MultiLayerNetwork = fitStraightline(ds)

    // Get the min and max x values, using Nd4j
    val preProcessor = new NormalizerMinMaxScaler
    preProcessor.fit(ds)
    val nSamples = 50
    val x = Nd4j.linspace(preProcessor.getMin.getInt(0), preProcessor.getMax.getInt(0), nSamples).reshape(nSamples, 1)
    val y = net.output(x)
    val modeloutput = new DataSet(x, y)
    DataSetList += modeloutput

    plotDataset(DataSetList.toArray) //Plot data and model fit.
  }

  /**
    * Fit a straight line using a neural network.
    *
    * @param ds The dataset to fit.
    * @return The network fitted to the data
    */
  private def fitStraightline(ds: DataSet): MultiLayerNetwork = {
    val seed: Int = 12345
    val iterations: Int = 1
    val nEpochs: Int = 200
    val learningRate: Double = 0.00001
    val numInputs: Int = 1
    val numOutputs: Int = 1
    //
    // Hook up one input to the one output.
    // The resulting model is a straight line.
    //
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
        .activation(Activation.IDENTITY)
        .build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation(Activation.IDENTITY).nIn(numOutputs).nOut(numOutputs)
        .build)
      .pretrain(false).backprop(true).build

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    for (i <- 0 until nEpochs) {
      net.fit(ds)
    }
    net
  }

  /**
    * Read a CSV file into a dataset.
    *
    * Use the correct constructor:
    * DataSet ds = new RecordReaderDataSetIterator(rr,batchSize);
    * returns the data as follows:
    * -----------INPUT-------------------
    * [
    *  [12.89, 22.70],
    *  [19.34, 20.47],
    *  [16.94,  6.08],
    *  [15.87,  8.42],
    *  [10.71, 26.18]
    * ]
    *
    * Which is not the way the framework likes its data.
    *
    * This one:
    * RecordReaderDataSetIterator(rr,batchSize, 1, 1, true);
    * returns
    * -----------INPUT-------------------
    * [12.89, 19.34, 16.94, 15.87, 10.71]
    * -----------------OUTPUT------------------
    * [22.70, 20.47,  6.08,  8.42, 26.18]
    *
    * This can be used as is for regression.
    */
  @throws[IOException]
  @throws[InterruptedException]
  private def readCSVDataset(filename: String): DataSet = {
    val batchSize = 1000
    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new File(filename)))

    val iter = new RecordReaderDataSetIterator(rr, batchSize, 1, 1, true)
    iter.next
  }

  /**
    * Generate an xy plot of the datasets provided.
    */
  private def plotDataset(DataSetList: Array[DataSet]) {
    val c: XYSeriesCollection = new XYSeriesCollection
    val dscounter = 1 //use to name the dataseries
    for (ds <- DataSetList) {
      val features: INDArray = ds.getFeatures
      val outputs: INDArray = ds.getLabels
      val nRows: Int = features.rows
      val series: XYSeries = new XYSeries("S" + dscounter)
      for (i <- 0 until nRows) {
        series.add(features.getDouble(i), outputs.getDouble(i))
      }
      c.addSeries(series)
    }

    val title = "title"
    val xAxisLabel = "xAxisLabel"
    val yAxisLabel = "yAxisLabel"
    val orientation = PlotOrientation.VERTICAL
    val legend = false
    val tooltips = false
    val urls = false
    val chart = ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls)
    val panel = new ChartPanel(chart)

    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Training Data")

    f.setVisible(true)
  }
}
