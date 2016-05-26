package org.deeplearning4j.examples.feedforward.mnist


import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
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
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory


/** A MLP applied to digit classification for MNIST. */
object MLPMnistTwoLayerExample {

    lazy val log = LoggerFactory.getLogger(MLPMnistTwoLayerExample.getClass)

    @throws[Exception]
    def main(args: Array[String]): Unit = {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val batchSize = 64
        val rngSeed = 123
        val numEpochs = 15
        val rate = 0.0015

        //Get the DataSetIterators:
        val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, rngSeed)
        val mnistTest: DataSetIterator  = new MnistDataSetIterator(batchSize, false, rngSeed)


        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .activation("relu")
            .weightInit(WeightInit.XAVIER)
            .learningRate(rate)
            .updater(Updater.NESTEROVS).momentum(0.98)
            .regularization(true).l2(rate * 0.005)
            .list()
            .layer(0, new DenseLayer.Builder()
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build())
            .layer(1,  new DenseLayer.Builder()
                    .nIn(500)
                    .nOut(100)
                    .build())
            .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(100)
                    .nOut(outputNum)
                    .build())
            .pretrain(false).backprop(true)
            .build()

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(5))

        log.info("Train model....")
        Seq.range(0, numEpochs).foreach { i =>
        	log.info("Epoch " + i)
            model.fit(mnistTrain)
        }


        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)
        while (mnistTest.hasNext){
            val next: DataSet = mnistTest.next()
            val output: INDArray = model.output(next.getFeatureMatrix)
            eval.eval(next.getLabels, output)
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
