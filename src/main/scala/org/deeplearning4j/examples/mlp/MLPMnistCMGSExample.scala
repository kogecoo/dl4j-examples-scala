package org.deeplearning4j.examples.mlp


import java.util.Collections
import java.util.Random

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.examples.mlp.sampleNetStructure.CMGSNet
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable


/**
 * Deep, Big, Simple Neural Nets Excel on Handwritten Digit Recognition
 * 2010 paper by Cireșan, Meier, Gambardella, and Schmidhuber
 * They achieved 99.65 accuracy
 */
object MLPMnistCMGSExample {

    lazy val log: Logger = LoggerFactory.getLogger(MLPMnistCMGSExample.getClass)


    def main(args: Array[String]): Unit = {

        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val numSamples = 60000
        val batchSize = 500
        val iterations = 50
        val seed = 123
        val listenerFreq = 10
        val splitTrainNum = (batchSize*.8).toInt

        val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
        val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]


        log.info("Load data....")
        val iter = new MnistDataSetIterator(batchSize,numSamples)

        log.info("Build model....")
        val model = new CMGSNet(numRows, numColumns, outputNum, seed, iterations).init()

        model.setListeners(Collections.singletonList[IterationListener](new ScoreIterationListener(listenerFreq)))

        log.info("Train model....")
        while(iter.hasNext) {
            val mnist = iter.next()
            val trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            val trainInput = trainTest.getTrain // get feature matrix and labels for training
            testInputBuilder += trainTest.getTest.getFeatureMatrix
            testLabelsBuilder += trainTest.getTest.getLabels
            model.fit(trainInput)
        }

        val testInput = testInputBuilder.result
        val testLabels = testLabelsBuilder.result

        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)
        testInput.zip(testLabels).foreach { case (input, label) =>
          val output: INDArray = model.output(input)
          eval.eval(label, output)
        }
        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
