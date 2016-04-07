package org.deeplearning4j.examples.unsupervised.autoencoder

import java.util.Collections

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{AutoEncoder, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory


/**
 * Created by agibsonccc on 9/11/14.
 */
object StackedAutoEncoderMnistExample {

    lazy val log = LoggerFactory.getLogger(StackedAutoEncoderMnistExample.getClass)

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
        val iter = new MnistDataSetIterator(batchSize,numSamples,true)

        log.info("Build model....")
        val conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(4)
           .layer(0, new AutoEncoder.Builder().nIn(numRows * numColumns).nOut(500)
                   .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                   .corruptionLevel(0.3)
                   .build())
                .layer(1, new AutoEncoder.Builder().nIn(500).nOut(250)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)

                        .build())
                .layer(2, new AutoEncoder.Builder().nIn(250).nOut(200)
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
                .build()

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Collections.singletonList[IterationListener](new ScoreIterationListener(listenerFreq)))

        log.info("Train model....")
        model.fit(iter); // achieves end to end pre-training

        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)

        val testIter = new MnistDataSetIterator(100,10000)
        while(testIter.hasNext) {
            val testMnist = testIter.next()
            val predict2 = model.output(testMnist.getFeatureMatrix)
            eval.eval(testMnist.getLabels, predict2)
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
