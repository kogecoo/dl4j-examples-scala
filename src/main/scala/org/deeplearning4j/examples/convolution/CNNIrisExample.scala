package org.deeplearning4j.examples.convolution

import java.util.Random

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

object CNNIrisExample {

  lazy val log = LoggerFactory.getLogger(CNNIrisExample.getClass)

  def main(args: Array[String]) = {


    val numRows = 2
    val numColumns = 2
    val nChannels = 1
    val outputNum = 3
    val numSamples = 150
    val batchSize = 110
    val iterations = 10
    val splitTrainNum = 100
    val seed = 123
    val listenerFreq = 1

    /**
      *Set a neural network configuration with multiple layers
      */

    log.info("Load data....")
    val irisIter = new IrisDataSetIterator(150, 150)
    val iris = irisIter.next()
    iris.normalizeZeroMeanZeroUnitVariance()
    Nd4j.shuffle(iris.getFeatureMatrix(), new Random(seed), 1)
    Nd4j.shuffle(iris.getLabels(),new Random(seed),1)
    val trainTest = iris.splitTestAndTrain(splitTrainNum, new Random(seed))

    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list(2)
      .layer(0, new ConvolutionLayer.Builder(Array(1, 1):_*)
        .nIn(nChannels)
        .nOut(1000)
        .activation("relu")
        .weightInit(WeightInit.RELU)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)

    new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)

    val conf: MultiLayerConfiguration = builder.build()

    log.info("Build model....")
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

    log.info("Train model....")
    model.fit(trainTest.getTrain())

    log.info("Evaluate weights....")
    model.getLayers.foreach { case (layer: org.deeplearning4j.nn.api.Layer) =>
      val w: INDArray = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
      log.info("Weights: " + w)
    }

    log.info("Evaluate model....")
    val eval: Evaluation[Nothing] = new Evaluation(outputNum)
    val output: INDArray = model.output(trainTest.getTest().getFeatureMatrix())
    eval.eval(trainTest.getTest().getLabels(), output)
    log.info(eval.stats())

    log.info("****************Example finished********************")
  }
}
