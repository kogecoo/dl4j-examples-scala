package org.deeplearning4j.examples.convolution

import java.util.Random

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.{DataSetIterator, MultipleEpochsIterator}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable


object LenetMnistExample {

  lazy val log: Logger = LoggerFactory.getLogger(LenetMnistExample.getClass)

  def main(args: Array[String]) = {
    Nd4j.dtype = DataBuffer.Type.DOUBLE
    Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
    Nd4j.ENFORCE_NUMERICAL_STABILITY = true

    val nChannels = 1
    val outputNum = 10
    val numSamples = 60000
    val batchSize = 500
    val iterations = 1
    val splitTrainNum = (batchSize*.8).toInt
    val seed = 123
    val listenerFreq = iterations/5
    val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
    val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]

    log.info("Load data....")
    val mnistIter: DataSetIterator = new MultipleEpochsIterator(5,new MnistDataSetIterator(batchSize,numSamples, true))

    log.info("Build model....")
    val builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true).l2(0.0005)
      .learningRate(0.01)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list(6)
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20).dropOut(0.5)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .build())
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        .nIn(20)
        .nOut(50)
        .stride(2,2)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .build())
      .layer(4, new DenseLayer.Builder().activation("tanh")
        .nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false);

    new ConvolutionLayerSetup(builder,28,28,1)

    val conf: MultiLayerConfiguration = builder.build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()

    log.info("Train model....")
    model.setListeners(new ScoreIterationListener(listenerFreq), new HistogramIterationListener(1))
    while(mnistIter.hasNext()) {
      val mnist = mnistIter.next()
      val trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
      val trainInput = trainTest.getTrain(); // get feature matrix and labels for training
      testInputBuilder += trainTest.getTest().getFeatureMatrix()
      testLabelsBuilder += trainTest.getTest().getLabels()

      model.fit(trainInput)
    }

    val testInput = testInputBuilder.result
    val testLabels = testLabelsBuilder.result

    log.info("Evaluate weights....")

    log.info("Evaluate model....")
    val eval = new Evaluation(outputNum)
    testInput.zip(testLabels).foreach { case (input, label) =>
      val output = model.output(input)
      eval.eval(label, output)
    }
    val output = model.output(testInput.head)
    eval.eval(testLabels.head, output)
    log.info(eval.stats())
    log.info("****************Example finished********************")


  }

}
