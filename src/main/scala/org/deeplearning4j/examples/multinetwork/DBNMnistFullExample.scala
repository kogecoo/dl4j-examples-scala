package org.deeplearning4j.examples.unsupervised.deepbelief

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.RBM
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.util.Collections

/**
 * Created by agibsonccc on 9/11/14.
 *
 * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
 */
object DBNMnistFullExample {

    lazy val log: Logger = LoggerFactory.getLogger(DBNMnistFullExample.getClass)

    def main(args: Array[String]): Unit = {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val numSamples = 60000
        val batchSize = 100
        val iterations = 10
        val seed = 123
        val listenerFreq = batchSize / 5

        log.info("Load data....")
        val iter: DataSetIterator = new MnistDataSetIterator(batchSize,numSamples,true)

        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(4)
           .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500)
                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                         .visibleUnit(RBM.VisibleUnit.BINARY)
                         .hiddenUnit(RBM.HiddenUnit.BINARY)
                         .build())
           .layer(1, new RBM.Builder().nIn(500).nOut(250)
                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                         .visibleUnit(RBM.VisibleUnit.BINARY)
                         .hiddenUnit(RBM.HiddenUnit.BINARY)
                         .build())
           .layer(2, new RBM.Builder().nIn(250).nOut(200)
                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                         .visibleUnit(RBM.VisibleUnit.BINARY)
                         .hiddenUnit(RBM.HiddenUnit.BINARY)
                         .build())
           .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                         .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
           .build()

        val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(listenerFreq).asInstanceOf[IterationListener])

        log.info("Train model....")
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)

        val testIter: DataSetIterator = new MnistDataSetIterator(100,10000)
        while (testIter.hasNext()) {
            val testMnist = testIter.next()
            val predict2: INDArray = model.output(testMnist.getFeatureMatrix())
            eval.eval(testMnist.getLabels(), predict2)
        }
        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
