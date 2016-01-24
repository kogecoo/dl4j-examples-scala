package org.deeplearning4j.examples.rbm

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._


object RBMIrisExample {

    val log = LoggerFactory.getLogger(RBMIrisExample.getClass)

    def main(args: Array[String])  = {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true

        val numRows = 4
        val numColumns = 1
        val outputNum = 3
        val numSamples = 150
        val batchSize = 150
        val iterations = 100
        val seed = 123
        val listenerFreq = iterations/5

        log.info("Load data....")
        val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)
        // Loads data into generator and format consumable for NN
        val iris: DataSet = iter.next()

        iris.scale()

        log.info("Build model....")
        val conf: NeuralNetConfiguration = new NeuralNetConfiguration.Builder()
                // Gaussian for visible; Rectified for hidden
                // Set contrastive divergence to 1
                .layer(new RBM.Builder()
                        .nIn(numRows * numColumns) // Input nodes
                        .nOut(outputNum) // Output nodes
                        .activation("tanh") // Activation function type
                        .weightInit(WeightInit.XAVIER) // Weight initialization
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .updater(Updater.NESTEROVS)
                        .build())
                .seed(seed) // Locks in weight initialization for tuning
                .learningRate(1e-1f) // Backprop step size
                .momentum(0.5) // Speed of modifying learning rate
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        // ^^ Calculates gradients
                .build()
        val model: Layer = LayerFactories.getFactory(conf.getLayer).create(conf)
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        log.info("Evaluate weights....")
        val w: INDArray = model.getParam(DefaultParamInitializer.WEIGHT_KEY)
        log.info("Weights: " + w)

        log.info("Train model....")
        model.fit(iris.getFeatureMatrix)

    }

    // A single layer learns features unsupervised.

}
