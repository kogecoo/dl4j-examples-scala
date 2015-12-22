package org.deeplearning4j.examples.mlp

import scala.collection.JavaConverters._

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory

object MLPBackpropIrisExample {

  lazy val log = LoggerFactory.getLogger(MLPBackpropIrisExample.getClass)

  def main(args: Array[String]) = {
    // Customizing params
    Nd4j.MAX_SLICES_TO_PRINT = 10
    Nd4j.MAX_ELEMENTS_PER_SLICE = 10

    val numInputs = 4
    val outputNum = 3
    val numSamples = 150
    val batchSize = 150
    val iterations = 100
    val seed: Long = 6L
    val listenerFreq = iterations/5

    log.info("Load data....")
    val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)

    log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)

      .learningRate(1e-3)
      .l1(0.3).regularization(true).l2(1e-3)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .list(3)
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
        .activation("tanh")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .nIn(2).nOut(outputNum).build())
      .backprop(true).pretrain(false)
      .build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

    log.info("Train model....")
    while (iter.hasNext()) {
      val iris: DataSet = iter.next()
      iris.normalizeZeroMeanZeroUnitVariance()
      model.fit(iris)
    }
    iter.reset()

    log.info("Evaluate weights....")
    model.getLayers().foreach { case (layer: org.deeplearning4j.nn.api.Layer) =>
      val w: INDArray  = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
      log.info("Weights: " + w)
    }


    log.info("Evaluate model....")
    val eval: Evaluation[Nothing] = new Evaluation(outputNum)
    val iterTest: DataSetIterator = new IrisDataSetIterator(numSamples, numSamples);
    val test: DataSet = iterTest.next()
    test.normalizeZeroMeanZeroUnitVariance()
    val output: INDArray = model.output(test.getFeatureMatrix())
    eval.eval(test.getLabels(), output)
    log.info(eval.stats())
    log.info("****************Example finished********************")
  }
}
