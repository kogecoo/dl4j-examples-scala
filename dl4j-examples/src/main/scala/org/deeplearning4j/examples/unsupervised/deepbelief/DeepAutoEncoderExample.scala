package org.deeplearning4j.examples.unsupervised.deepbelief

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

/**
  * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
  *
  * @author Adam Gibson
  */
object DeepAutoEncoderExample {
  private val log: Logger = LoggerFactory.getLogger(DeepAutoEncoderExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    val numRows = 28
    val numColumns = 28
    val seed = 123
    val numSamples = MnistDataFetcher.NUM_EXAMPLES
    val batchSize = 1000
    val iterations = 1
    val listenerFreq = iterations / 5

    log.info("Load data....")
    val iter = new MnistDataSetIterator(batchSize, numSamples, true)

    log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .list
      .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(1, new RBM.Builder().nIn(1000).nOut(500).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build).layer(2, new RBM.Builder().nIn(500).nOut(250).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(3, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(4, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build) //encoding stops
      .layer(5, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build) //decoding starts
      .layer(6, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(7, new RBM.Builder().nIn(250).nOut(500).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(8, new RBM.Builder().nIn(500).nOut(1000).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(1000).nOut(numRows * numColumns).build)
      .pretrain(true).backprop(true)
      .build

    val model = new MultiLayerNetwork(conf)
    model.init()

    model.setListeners(new ScoreIterationListener(listenerFreq))

    log.info("Train model....")
    while (iter.hasNext) {
      val next = iter.next
      model.fit(new DataSet(next.getFeatureMatrix, next.getFeatureMatrix))
    }
  }
}
