package org.deeplearning4j.examples.feedforward.classification

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.io.File

/**
  * "Moon" Data Classification Example
  *
  * Based on the data from Jason Baldridge:
  * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
  *
  * @author Josh Patterson
  * @author Alex Black (added plots)
  *
  */
object MLPClassifierMoon {

  @throws[Exception]
  def main(args: Array[String]) {
    val seed = 123
    val learningRate = 0.005
    val batchSize = 50
    val nEpochs = 100

    val numInputs = 2
    val numOutputs = 2
    val numHiddenNodes = 20

    val filenameTrain = new ClassPathResource("/classification/moon_data_train.csv").getFile.getPath
    val filenameTest = new ClassPathResource("/classification/moon_data_eval.csv").getFile.getPath

    //Load the training data:
    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new File(filenameTrain)))
    var trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2)

    //Load the test/evaluation data:
    val rrTest = new CSVRecordReader
    rrTest.initialize(new FileSplit(new File(filenameTest)))
    var testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2)

    //log.info("Build model....");
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(1)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .build)
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .nIn(numHiddenNodes)
        .nOut(numOutputs)
        .build)
      .pretrain(false).backprop(true)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100)) //Print score every 100 parameter updates

    for (n <- 0 until nEpochs) {
      model.fit(trainIter)
    }

    println("Evaluate model....")
    val eval = new Evaluation(numOutputs)
    while (testIter.hasNext) {
      val t = testIter.next
      val features = t.getFeatureMatrix
      val labels = t.getLabels
      val predicted = model.output(features, false)

      eval.eval(labels, predicted)
    }

    //Print the evaluation statistics
    println(eval.stats)

    //------------------------------------------------------------------------------------
    //Training is complete. Code that follows is for plotting the data & predictions only

    //Plot the data
    val xMin = -1.5
    val xMax = 2.5
    val yMin = -1
    val yMax = 1.5

    //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
    val nPointsPerAxis: Int = 100
    val evalPoints = for {
      i <- 0 until nPointsPerAxis
      j <- 0 until nPointsPerAxis
    } yield {
      val x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin
      val y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin
      Array(x, y)
    }
    val allXYPoints = Nd4j.create(evalPoints.toArray)
    val predictionsAtXYPoints: INDArray = model.output(allXYPoints)

    //Get all of the training data in a single array, and plot it:
    rr.initialize(new FileSplit(new File("dl4j-examples/src/main/resources/classification/moon_data_train.csv")))
    rr.reset()
    val nTrainPoints = 2000
    trainIter = new RecordReaderDataSetIterator(rr, nTrainPoints, 0, 2)
    var ds = trainIter.next
    PlotUtil.plotTrainingData(ds.getFeatures, ds.getLabels, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)


    //Get test data, run the test data through the network to generate predictions, and plot those predictions:
    rrTest.initialize(new FileSplit(new File("dl4j-examples/src/main/resources/classification/moon_data_eval.csv")))
    rrTest.reset()
    val nTestPoints = 1000
    testIter = new RecordReaderDataSetIterator(rrTest, nTestPoints, 0, 2)
    ds = testIter.next
    val testPredicted = model.output(ds.getFeatures)
    PlotUtil.plotTestData(ds.getFeatures, ds.getLabels, testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

    println("****************Example finished********************")
  }
}
