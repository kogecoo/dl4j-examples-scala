package org.deeplearning4j.examples.rbm

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.nn.api.{Layer, OptimizationAlgorithm}
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.conf.layers.RBM.{HiddenUnit, VisibleUnit}
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory


object RBMIrisExample {

    val log = LoggerFactory.getLogger(RBMIrisExample.getClass)

    def main(args: Array[String])  = {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true

        val numRows = 4
        val numColumns = 1
        val outputNum = 10
        val numSamples = 150
        val batchSize = 150
        val iterations = 100
        val seed = 123
        val listenerFreq = iterations/2

        log.info("Load data....")
        val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)
        // Loads data into generator and format consumable for NN
        val iris: DataSet = iter.next()

        iris.normalizeZeroMeanZeroUnitVariance()

        log.info("Build model....")
        val conf: NeuralNetConfiguration = new NeuralNetConfiguration.Builder().regularization(true)
                  .miniBatch(true)
                // Gaussian for visible; Rectified for hidden
                // Set contrastive divergence to 1
                .layer(new RBM.Builder().l2(1e-1).l1(1e-3)
                        .nIn(numRows * numColumns) // Input nodes
                        .nOut(outputNum) // Output nodes
                        .activation("relu") // Activation function type
                        .weightInit(WeightInit.RELU) // Weight initialization
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).k(3)
                        .hiddenUnit(HiddenUnit.RECTIFIED).visibleUnit(VisibleUnit.GAUSSIAN)
                        .updater(Updater.ADAGRAD).gradientNormalization(GradientNormalization.ClipL2PerLayer)
                        .build())
                .seed(seed) // Locks in weight initialization for tuning
                .iterations(iterations)
                .learningRate(1e-3) // Backprop step size
                // Speed of modifying learning rate
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                        // ^^ Calculates gradients
                .build()
        val model: Layer = LayerFactories.getFactory(conf.getLayer).create(conf)
        model.setListeners(new ScoreIterationListener(listenerFreq))

        log.info("Evaluate weights....")
        val w: INDArray = model.getParam(DefaultParamInitializer.WEIGHT_KEY)
        log.info("Weights: " + w)
        log.info("Scaling the dataset")
        iris.scale()
        log.info("Train model....")
        (0 until 20).foreach { i =>
            log.info("Epoch " + i + ":")
            model.fit(iris.getFeatureMatrix)
        }

    }

    // A single layer learns features unsupervised.

}
