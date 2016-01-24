package org.deeplearning4j.examples.test

import java.util.Random

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._


object Test {
    val LOGGER = LoggerFactory.getLogger(Test.getClass)

    def main(args: Array[String]) {
        val numRows = 4
        val numColumns = 1
        val outputNum = 3
        val numSamples = 150
        val batchSize = 150
        val iterations = 10
        val splitTrainNum = (batchSize * .8).toInt
        val seed = 123
        val listenerFreq = iterations/5
        Nd4j.getRandom.setSeed(seed)
        LOGGER.info("Load data....")
        val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)
        val iris: DataSet = iter.next()
        iris.normalizeZeroMeanZeroUnitVariance()

        LOGGER.info("Split data....")
        val testAndTrain: SplitTestAndTrain = iris.splitTestAndTrain(splitTrainNum, new Random(seed))
        val train: DataSet = testAndTrain.getTrain
        val test: DataSet = testAndTrain.getTest
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true

        LOGGER.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(1e-6f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop method (calculate the gradients)
                .l1(1e-1).regularization(true).l2(2e-4)
                .useDropConnect(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                                .nIn(numRows * numColumns) // # input nodes
                                .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                                .weightInit(WeightInit.XAVIER) // Weight initialization method
                                .k(1) // # contrastive divergence iterations
                                .activation("relu") // Activation function type
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                                .updater(Updater.ADAGRAD)
                                .dropOut(0.5)
                                .build()
                ) // NN layer type
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(3) // # input nodes
                                .nOut(outputNum) // # output nodes
                                .activation("softmax")
                                .build()
                ) // NN layer type
                .build()
        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        LOGGER.info("Train model....")
        model.fit(train)

//        for(int i=0; i<10; i++){
//            LOGGER.info("Output: {}", model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST).getRow(0))
//        }
    }

}
