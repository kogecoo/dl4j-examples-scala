package org.deeplearning4j.examples.convolution

import java.util._

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuilder

object CNNMnistExample {

    lazy val log = LoggerFactory.getLogger(CNNMnistExample.getClass)

    def main(args: Array[String]) = {

        val numRows = 28
        val numColumns = 28
        val nChannels = 1
        val outputNum = 10
        val numSamples = 2000
        val batchSize = 500
        val iterations = 10
        val splitTrainNum = (batchSize*.8).toInt
        val seed = 123
        val listenerFreq = iterations/5
        val testInputBuilder = ArrayBuilder.make[INDArray]
        val testLabelsBuilder = ArrayBuilder.make[INDArray]

        log.info("Load data....")
        val mnistIter: DataSetIterator = new MnistDataSetIterator(batchSize,numSamples, true)

        log.info("Build model....")
      val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
        new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)

        val conf: MultiLayerConfiguration = builder.build()

        val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
        model.init()

        log.info("Train model....")
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)
        while(mnistIter.hasNext()) {
            val mnist = mnistIter.next()
            val trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            val trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInputBuilder += trainTest.getTest().getFeatureMatrix()
            testLabelsBuilder += trainTest.getTest().getLabels()
            model.fit(trainInput)
        }

        val testInput: Array[INDArray] = testInputBuilder.result
        val testLabels: Array[INDArray] = testLabelsBuilder.result

        log.info("Evaluate weights....")

        log.info("Evaluate model....")
        val eval: Evaluation = new Evaluation(outputNum)
        testInput.zip(testLabels).foreach { case (input, label) =>
          val output: INDArray = model.output(input)
          eval.eval(label, output)
        }
        val output: INDArray = model.output(testInput.head)
        eval.eval(testLabels.head, output)
        log.info(eval.stats())
        log.info("****************Example finished********************")

    }
}
