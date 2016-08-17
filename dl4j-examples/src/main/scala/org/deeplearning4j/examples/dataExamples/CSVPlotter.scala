package org.deeplearning4j.examples.dataExamples

import java.io.File
import java.io.IOException
import java.util.ArrayList

import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.WindowConstants

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYDataset
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.springframework.core.io.ClassPathResource

import scala.collection.JavaConverters._
/**
 * Read a csv file. Fit and plot the data using Deeplearning4J.
 *
 * @author Robert Altena
 */
class CSVPlotter {

	  @throws[IOException]
  	@throws[InterruptedException]
    def main(args: Array[String]) {
			val filename = new ClassPathResource("/dataExamples/CSVPlotData.csv").getFile.getPath
    	val ds = readCSVDataset(filename)

    	val dataSetList = new ArrayList[DataSet]()
    	dataSetList.add(ds)

    	plotDataset(dataSetList); //Plot the data, make sure we have the right data.

    	val net: MultiLayerNetwork = fitStraightline(ds)

    	// Get the min and max x values, using Nd4j
    	val preProcessor: NormalizerMinMaxScaler = new NormalizerMinMaxScaler()
    	preProcessor.fit(ds)
        val nSamples = 50
        val x = Nd4j.linspace(preProcessor.getMin.getInt(0),preProcessor.getMax.getInt(0),nSamples).reshape(nSamples, 1)
        val y = net.output(x)
        val modeloutput = new DataSet(x,y)
        dataSetList.add(modeloutput)

    	plotDataset(dataSetList);    //Plot data and model fit.
    }

	/**
	 * Fit a straight line using a neural network.
	 * @param ds The dataset to fit.
	 * @return The network fitted to the data
	 */
	private def fitStraightline(ds: DataSet): MultiLayerNetwork = {
      val seed = 12345
      val iterations = 1
      val nEpochs = 200
      val learningRate = 0.00001
      val numInputs = 1
	    val numOutputs = 1

	    //
	    // Hook up one input to the one output.
	    // The resulting model is a straight line.
	    //
		  val conf = new  NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
                        .activation("identity")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(numOutputs).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()

      val net = new MultiLayerNetwork(conf)
      net.init()
	    net.setListeners(new ScoreIterationListener(1))

			(0 until nEpochs).foreach { i => net.fit(ds) }

	    net
	}

    /**
     * Read a CSV file into a dataset.
     *
     * Use the correct constructor:
     * DataSet ds = new RecordReaderDataSetIterator(rr,batchSize)
     * returns the data as follows:
     * ---------------INPUT------------------
     * [ [12.89, 22.70],
     * [19.34, 20.47],
     * [16.94,  6.08],
     *  [15.87,  8.42],
     *  [10.71, 26.18] ]
     *
     *  Which is not the way the framework likes its data.
     *
     *  This one:
     *   RecordReaderDataSetIterator(rr,batchSize, 1, 1, true)
     *   returns
		 *   ---------------INPUT------------------
     * [12.89, 19.34, 16.94, 15.87, 10.71]
     * ---------------OUTPUT------------------
     * [22.70, 20.47,  6.08,  8.42, 26.18]
     *
     *  This can be used as is for regression.
     */
  @throws[IOException]
	@throws[InterruptedException]
	private def readCSVDataset(filename: String): DataSet = {
		val batchSize = 1000
		val rr = new CSVRecordReader()
		rr.initialize(new FileSplit(new File(filename)))

		val iter =  new RecordReaderDataSetIterator(rr,batchSize, 1, 1, true)
		iter.next()
	}

	/**
	 * Generate an xy plot of the datasets provided.
	 */
	private def plotDataset(dataSetList: java.util.List[DataSet] ){

		val c = new XYSeriesCollection()

		val dscounter = 1; //use to name the dataseries
		dataSetList.asScala.foreach { ds: DataSet =>
			val features = ds.getFeatures
			val outputs= ds.getLabels

			val nRows = features.rows()
			val series = new XYSeries("S" + dscounter)
			(0 until nRows).foreach { i =>
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
		val chart: JFreeChart = ChartFactory.createScatterPlot(title , xAxisLabel, yAxisLabel, c, orientation , legend , tooltips , urls)
    	val panel: JPanel = new ChartPanel(chart)

    	 val f = new JFrame()
    	 f.add(panel)
    	 f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
         f.pack()
         f.setTitle("Training Data")

         f.setVisible(true)
	}
}
