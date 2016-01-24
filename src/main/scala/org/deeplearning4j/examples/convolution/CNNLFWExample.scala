package org.deeplearning4j.examples.convolution

import java.util.{Collections, Random}

import org.canova.image.loader.LFWLoader
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable


/**
 * Labeled Faces in the Wild is a data set created by Erik Learned-Miller, Gary Huang, Aruni RoyChowdhury,
 * Haoxiang Li, Gang Hua. This is used to study unconstrained face recognition. There are over 13K images.
 * Each face has been labeled with the name of the person pictured.
 *
 * 5749 unique classes (different people)
 * 1680 people have 2+ photos
 *
 * References:
 * General information is at http://vis-www.cs.umass.edu/lfw/. Architecture partially based on DeepFace:
 * http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
 *
 * Note: this is a sparse dataset with only 1 example for many of the faces; thus, performance is low.
 * Ideally train on a larger dataset like celebs to get params and/or generate variations of the image examples.
 *
 * Currently set to only use the subset images, names starting with A.
 * Switch to NUM_LABELS & NUM_IMAGES and set subset to false to use full dataset.
 */

object CNNLFWExample {
    lazy val log: Logger = LoggerFactory.getLogger(CNNMnistExample.getClass)

    def main(args: Array[String]) = {
        val numRows = 40
        val numColumns = 40
        val nChannels = 3
        val outputNum = LFWLoader.NUM_LABELS
        val numSamples = 1000 // LFWLoader.NUM_IMAGES
        val useSubset = false

        val batchSize = 200 // numSamples/10
        val iterations = 5
        val splitTrainNum = (batchSize*.8).toInt
        val seed = 123
        val listenerFreq = iterations/5
        val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
        val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]


        log.info("Load data.....")
        val lfw: DataSetIterator = new LFWDataSetIterator(batchSize, numSamples, Array[Int](numRows, numColumns, nChannels), outputNum, false, new Random(seed))

        log.info("Build model....")
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation("relu")
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.01)
                .momentum(0.9)
                .regularization(true)
                .updater(Updater.ADAGRAD)
                .useDropConnect(true)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(4, 4)
                        .name("cnn1")
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .name("pool1")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn2")
                        .stride(1,1)
                        .nOut(40)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .name("pool2")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(60)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .name("pool3")
                        .build())
                .layer(2, new ConvolutionLayer.Builder(2, 2)
                        .name("cnn3")
                        .stride(1,1)
                        .nOut(80)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(160)
                        .dropOut(0.5)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)

        val model = new MultiLayerNetwork(builder.build())
        model.init()

        log.info("Train model....")
        model.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq).asInstanceOf[IterationListener]))


        while(lfw.hasNext) {
            val next: DataSet = lfw.next()
            next.scale()
            val trainTest = next.splitTestAndTrain(splitTrainNum, new Random(seed))  // train set that is the result
            val trainInput = trainTest.getTrain  // get feature matrix and labels for training
            testInputBuilder += trainTest.getTest.getFeatureMatrix
            testLabelsBuilder += trainTest.getTest.getLabels
            model.fit(trainInput)
        }

        val testInput = testInputBuilder.result
        val testLabels = testLabelsBuilder.result

        log.info("Evaluate model....")
        val eval = new Evaluation(lfw.getLabels)
        testInput.zip(testLabels).foreach { case (input, label) =>
          val output: INDArray = model.output(input)
          eval.eval(label, output)
        }
        val output: INDArray = model.output(testInput.head)
        eval.eval(testLabels.head, output)
        log.info(eval.stats())
        log.info("****************Example finished********************")
    }

}
