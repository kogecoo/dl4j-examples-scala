package org.deeplearning4j.examples.feedforward.mnist

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory


/**A Simple MLP applied to digit classification for MNIST.
  */
object MLPMnistSingleLayerExample {

    lazy val log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.getClass)

    def main(args: Array[String]) = {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val batchSize = 128
        val rngSeed = 123
        val numEpochs = 15

        //Get the DataSetIterators:
        val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)

        log.info("Build model....")
        val conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
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
                .pretrain(false).backprop(true)
                .build()

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(1))

        log.info("Train model....")
        (0 until numEpochs).foreach(_  => model.fit(mnistTrain))

        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)
        while (mnistTest.hasNext) {
            val next = mnistTest.next()
            val output = model.output(next.getFeatureMatrix)
            eval.eval(next.getLabels, output)
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
