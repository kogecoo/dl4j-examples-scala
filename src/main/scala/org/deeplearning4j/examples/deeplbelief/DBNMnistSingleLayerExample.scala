package org.deeplearning4j.examples.deepbelief


import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration }
import org.deeplearning4j.nn.conf.layers.{ OutputLayer, RBM }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.util.Arrays
import scala.collection.JavaConverters._


object DBNMnistSingleLayerExample {

    lazy val log = LoggerFactory.getLogger(DBNMnistSingleLayerExample.getClass)

    def main(args: Array[String]) = {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true
        val numRows = 28
        val numColumns = 28
        val outputNum = 10
        val numSamples = 500
        val batchSize = 500
        val iterations = 10
        val seed = 123
        val listenerFreq = iterations/5

        log.info("Load data....")
        val iter: DataSetIterator = new MnistDataSetIterator(batchSize, numSamples, true)
        log.info("Build model....")
        val conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .iterations(iterations).constrainGradientToUnitNorm(true)
                .learningRate(1e-1f)
                .list(2)
                .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(500).activation("relu")
                        .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nIn(500).nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build()

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        log.info("Train model....")
        while(iter.hasNext()) {
            val mnist: DataSet = iter.next()
            model.fit(mnist)
        }
        iter.reset()

        log.info("Evaluate weights....")
        log.info("Evaluate model....")
        val eval: Evaluation = new Evaluation(outputNum)
        while(iter.hasNext()) {
            val testData: DataSet = iter.next()
            val predict2: INDArray = model.output(testData.getFeatureMatrix())
            eval.eval(testData.getLabels(), predict2)
        }

        log.info(eval.stats())
        log.info("****************Example finished********************")

    }

}
