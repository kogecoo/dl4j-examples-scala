package org.deeplearning4j.examples.convolution

import java.util.Random

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuilder

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

/**
  * @author Adam Gibson
  */
object CNNLFWDataSet {
  lazy val log: Logger = LoggerFactory.getLogger(CNNMnistExample.getClass)

  def main(args: Array[String]) = {
    val numRows = 28
    val numColumns = 28
    val nChannels = 1
    val outputNum = 5749
    val numSamples = 2000
    val batchSize = 500
    val iterations = 10
    val splitTrainNum = (batchSize*.8).toInt
    val seed = 123
    val listenerFreq = iterations/5
    val testInputBuilder = ArrayBuilder.make[INDArray]
    val testLabelsBuilder = ArrayBuilder.make[INDArray]


    log.info("Load data.....")
    val lfw: DataSetIterator = new LFWDataSetIterator(batchSize, numSamples)

    log.info("Build model....")
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list(3)
      .layer(0, new ConvolutionLayer.Builder(10, 10)
        .nIn(nChannels)
        .nOut(6)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false);
    new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)

    val model = new MultiLayerNetwork(builder.build())
    model.init()

    log.info("Train model....")
    model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)


    while(lfw.hasNext()) {
      val next: DataSet = lfw.next()
      next.scale()
      val trainTest = next.splitTestAndTrain(splitTrainNum, new Random(seed))  // train set that is the result
      val trainInput = trainTest.getTrain()  // get feature matrix and labels for training
      testInputBuilder += trainTest.getTest().getFeatureMatrix()
      testLabelsBuilder += trainTest.getTest().getLabels()
      model.fit(trainInput)
    }

    val testInput = testInputBuilder.result
    val testLabels = testLabelsBuilder.result

    log.info("Evaluate model....")
    val eval: Evaluation[Nothing] = new Evaluation(outputNum)
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
