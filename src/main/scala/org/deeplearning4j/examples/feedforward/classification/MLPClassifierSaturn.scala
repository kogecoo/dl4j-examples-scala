package org.deeplearning4j.examples.feedforward.classification

import java.io.File

import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 * "Saturn" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * 	https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
object MLPClassifierSaturn {


    def main(args: Array[String]): Unit = {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true
        val batchSize = 50
        val seed = 123
        val learningRate = 0.005
        //Number of epochs (full passes of the data)
        val nEpochs = 30

        val numInputs = 2
        val numOutputs = 2
        val numHiddenNodes = 20

        //Load the training data:
        val rr: RecordReader = new CSVRecordReader()
        rr.initialize(new FileSplit(new File("src/main/resources/classification/saturn_data_train.csv")))
        val trainIter: DataSetIterator = new RecordReaderDataSetIterator(rr,batchSize,0,2)

        //Load the test/evaluation data:
        val rrTest: RecordReader = new CSVRecordReader()
        rrTest.initialize(new FileSplit(new File("src/main/resources/classification/saturn_data_eval.csv")))
        val testIter: DataSetIterator = new RecordReaderDataSetIterator(rrTest,batchSize,0,2)

        //log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build()


        val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(1))

        (0 until nEpochs).foreach { _ => model.fit( trainIter ) }

        System.out.println("Evaluate model....")
        val eval = new Evaluation(numOutputs)
        while(testIter.hasNext()){
            val t = testIter.next()
            val features = t.getFeatureMatrix()
            val lables = t.getLabels()
            val predicted = model.output(features,false)

            eval.eval(lables, predicted)

        }


        System.out.println(eval.stats())
        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only


        val xMin = -15
        val xMax = 15
        val yMin = -15
        val yMax = 15

        //Let's evaluate the predictions at every point in the x/y input space, and plot this in the background
        val nPointsPerAxis = 100
        val evalPoints: Array[Array[Double]] = (0 until nPointsPerAxis).toArray.flatMap { i =>
            (0 until nPointsPerAxis).toArray.map { j =>
                val x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin
                val y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin
                Array[Double](x, y)
            }
        }

        val allXYPoints: INDArray = Nd4j.create(evalPoints)
        val predictionsAtXYPoints: INDArray = model.output(allXYPoints)

        //Get all of the training data in a single array, and plot it:
        rr.initialize(new FileSplit(new File("src/main/resources/classification/saturn_data_train.csv")))
        rr.reset()
        val nTrainPoints = 500
        val rrTrainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,0,2)
        val ds1: DataSet = rrTrainIter.next()
        PlotUtil.plotTrainingData(ds1.getFeatures(), ds1.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis)


        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
        rrTest.initialize(new FileSplit(new File("src/main/resources/classification/saturn_data_eval.csv")))
        rrTest.reset()
        val nTestPoints = 100
        val rrTestIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,0,2)
        val ds2
        = rrTestIter.next()
        val testPredicted = model.output(ds2
          .getFeatures())
        PlotUtil.plotTestData(ds2
          .getFeatures(), ds2
          .getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis)

        System.out.println("****************Example finished********************")
    }

}