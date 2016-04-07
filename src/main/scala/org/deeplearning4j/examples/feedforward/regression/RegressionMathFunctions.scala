package org.deeplearning4j.examples.feedforward.regression

import java.util.{Collections, List, Random}
import javax.swing._

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.examples.feedforward.regression.function._
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.conf.layers.{DenseLayer, GravesLSTM, OutputLayer, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.{XYSeries, XYSeriesCollection}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
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
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    val batchSize = 100
    //Network learning rate
    val learningRate = 0.01
    val rng: Random = new Random(seed)
    val numInputs = 1
    val numOutputs = 1


    def main(args: Array[String]): Unit = {

        //Switch these two options to do different functions with different networks
        val fn: MathFunction = new SinXDivXMathFunction()
        val conf: MultiLayerConfiguration = getDeepDenseLayerNetworkConfiguration()

        //Generate the training data
        val x: INDArray = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1)
        val iterator: DataSetIterator = getTrainingData(x,fn,batchSize,rng)

        //Create the network
        val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
        net.init()
        net.setListeners(new ScoreIterationListener(1))


        //Train the network on the full data set, and evaluate in periodically
        val networkPredictions: Array[INDArray] = Array.fill[INDArray](nEpochs/ plotFrequency)(null)
        (0 until nEpochs).foreach { i =>
            iterator.reset()
            net.fit(iterator)
            if((i+1) % plotFrequency == 0) networkPredictions(i/ plotFrequency) = net.output(x, false)
        }

        //Plot the target data and the network predictions
        plot(fn,x,fn.getFunctionValues(x),networkPredictions)
    }

    def getLSTMNetworkConfiguration(): MultiLayerConfiguration = {
        val numHiddenNodes = 20
        new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(iterations)
            .learningRate(learningRate)
            .weightInit(WeightInit.DISTRIBUTION)
            .rmsDecay(0.95)
            .seed(seed)
            .regularization(true)
            .l2(0.001)
            .list(3)
            .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .updater(Updater.RMSPROP)
                    .activation("softsign")
                    .dist(new UniformDistribution(-0.08, 0.08)).build())
            .layer(1, new GravesLSTM.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .updater(Updater.RMSPROP)
                    .activation("softsign")
                    .dist(new UniformDistribution(-0.08, 0.08)).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation("identity")
                    .updater(Updater.RMSPROP)
                    .nIn(numHiddenNodes).nOut(numOutputs)
                    .dist(new UniformDistribution(-0.08, 0.08)).build())
            .pretrain(false).backprop(true)
            .build()
    }

    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    def getDeepDenseLayerNetworkConfiguration(): MultiLayerConfiguration = {
        val numHiddenNodes = 50
        new NeuralNetConfiguration.Builder()
              .seed(seed)
              .iterations(iterations)
              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
              .learningRate(learningRate)
              .weightInit(WeightInit.XAVIER)
              .updater(Updater.NESTEROVS).momentum(0.9)
              .list(3)
              .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                      .activation("tanh")
                      .build())
              .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                      .activation("tanh")
                      .build())
              .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                      .activation("identity")
                      .nIn(numHiddenNodes).nOut(numOutputs).build())
              .pretrain(false).backprop(true).build()
    }

    /** Returns the network configuration, 1 hidden DenseLayer of size 20.
     */
    def getSimpleDenseLayerNetworkConfiguration(): MultiLayerConfiguration = {
        val numHiddenNodes = 20
        new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list(2)
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                    .activation("tanh")
                    .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation("identity")
                    .nIn(numHiddenNodes).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
    }

    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    def getTrainingData(x: INDArray, function: MathFunction, batchSize: Int, rng: Random): DataSetIterator = {
        val y = function.getFunctionValues(x)
        val allData: DataSet = new DataSet(x,y)

        val list: List[DataSet] = allData.asList()
        Collections.shuffle(list,rng)
        new ListDataSetIterator(list,batchSize)
    }

    //Plot the data
    def plot(function: MathFunction, x: INDArray, y: INDArray, predicted: Array[INDArray]): Unit = {
        val dataSet: XYSeriesCollection = new XYSeriesCollection()
        addSeries(dataSet,x,y,"True Function (Labels)")

        predicted.indices.foreach { i =>
            addSeries(dataSet,x,predicted(i),String.valueOf(i))
        }

        val chart: JFreeChart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.getName(),      // chart title
                "X",                        // x axis label
                function.getName() + "(X)", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        )

        val panel: ChartPanel = new ChartPanel(chart)

        val f: JFrame = new JFrame()
        f.add(panel)
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        f.pack()

        f.setVisible(true)
    }

    def addSeries(dataSet: XYSeriesCollection, x: INDArray, y: INDArray, label: String): Unit = {
        val xd: Array[Double] = x.data().asDouble()
        val yd: Array[Double] = y.data().asDouble()
        val s: XYSeries = new XYSeries(label)
        xd.indices.foreach { j => s.add(xd(j),yd(j)) }
        dataSet.addSeries(s)
    }
}
