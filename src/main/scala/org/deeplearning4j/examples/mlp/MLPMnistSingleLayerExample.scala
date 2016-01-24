package org.deeplearning4j.examples.mlp

import java.util._

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable


object MLPMnistSingleLayerExample {

    lazy val log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.getClass)

    def main(args: Array[String]) = {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val numSamples = 1000
        val batchSize = 500
        val iterations = 10
        val seed = 123
        val listenerFreq = iterations/5
        val splitTrainNum = (batchSize*.8).toInt
        val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
        val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]


        log.info("Load data....")
        val mnistIter: DataSetIterator = new MnistDataSetIterator(batchSize, numSamples, true)
        log.info("Build model....")
        val conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .learningRate(1e-1f)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .useDropConnect(true)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                  .nIn(numRows*numColumns)
                  .nOut(1000)
                  .activation("relu")
                  .weightInit(WeightInit.XAVIER)
                  .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                  .nIn(1000)
                  .nOut(outputNum)
                  .activation("softmax")
                  .weightInit(WeightInit.XAVIER)
                  .build())
                .build()

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        log.info("Train model....")
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)
        while(mnistIter.hasNext) {
            val mnist = mnistIter.next()
            val trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed))
            // train set that is the result
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
            val output = model.output(input)
            eval.eval(label, output)
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
