package org.deeplearning4j.examples.convolution

import org.deeplearning4j.datasets.fetchers.LFWDataFetcher
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration }
import org.deeplearning4j.nn.conf.layers.{ ConvolutionLayer, OutputLayer, SubsamplingLayer }
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.preprocessor.{ CnnToFeedForwardPreProcessor, FeedForwardToCnnPreProcessor }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.util.Arrays
import scala.collection.JavaConverters._

/**
 * @author Adam Gibson
 */
object LFWDataSet {
    lazy val log: Logger = LoggerFactory.getLogger(CNNMnistExample.getClass)

    def main(args: Array[String]) = {
        val lfw: DataSetIterator = new LFWDataSetIterator(28,28)

        val numRows = 28
        val numColumns = 28
        val nChannels = 1
        val outputNum = 5749
        val numSamples = 2000
        val batchSize = 500
        val iterations = 10
        val splitTrainNum = (batchSize*.8).toInt
        val seed = 123
        val listenerFreq = iterations/5


        log.info("Build model....")
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(nChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)

        val model = new MultiLayerNetwork(builder.build())
        model.init()

        log.info("Train model....")
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)


        while(lfw.hasNext()) {
            val next: DataSet = lfw.next()
            next.scale()
            model.fit(next)
        }
    }

}
