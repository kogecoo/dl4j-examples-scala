package org.deeplearning4j.examples.convolution

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}


object LenetMnistExample {

    lazy val log: Logger = LoggerFactory.getLogger(LenetMnistExample.getClass)

    def main(args: Array[String]) = {
        Nd4j.dtype = DataBuffer.Type.DOUBLE
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true

        val nChannels = 1
        val outputNum = 10
        val batchSize = 64
        val nEpochs = 10
        val iterations = 1
        val seed = 123

        log.info("Load data....")
        val mnistTrain = new MnistDataSetIterator(batchSize,true,12345)
        val mnistTest = new MnistDataSetIterator(batchSize,false,12345)

        log.info("Build model....")
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)

        new ConvolutionLayerSetup(builder,28,28,1)

        val conf: MultiLayerConfiguration = builder.build()

        val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
        model.init()

        log.info("Train model....")
        model.setListeners(new ScoreIterationListener(1))
        (0 until nEpochs).foreach { i =>
            model.fit(mnistTrain)
            log.info("*** Completed epoch {} ***", i)

            log.info("Evaluate model....")
            val eval = new Evaluation(outputNum)
            while(mnistTest.hasNext){
                val ds = mnistTest.next()
                val output = model.output(ds.getFeatureMatrix())
                eval.eval(ds.getLabels(), output)
            }
            log.info(eval.stats())
            mnistTest.reset()
        }
        log.info("****************Example finished********************")
    }

}
