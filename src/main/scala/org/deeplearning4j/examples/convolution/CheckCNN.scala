package org.deeplearning4j.examples.convolution

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.util.FeatureUtil

import java.util.Arrays
import scala.collection.JavaConverters._


/**
 * Created by nyghtowl on 10/28/15.
 */
object CheckCNN {
    def main(args: Array[String]): Unit = {
        val nChannels = 1
        val nSamples = 10
        val data = Nd4j.rand(nSamples,300)
        val labels = FeatureUtil.toOutcomeMatrix(Array(0,0,0,1,1,1,2,2,2,2), 3)
        val d = new DataSet(data,labels)
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(10)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10,10)
                        .stride(2,2)
                        .nIn(nChannels)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(
                        SubsamplingLayer.PoolingType.MAX, Array(2, 2))
                        .stride(2,2)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true)
                .pretrain(false)

        new ConvolutionLayerSetup(builder, 10, 30,
                nChannels);    // FIXME: what does this do?

        val conf: MultiLayerConfiguration = builder.build()

        val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(5)).asJava)
        model.fit(d)
    }

}
