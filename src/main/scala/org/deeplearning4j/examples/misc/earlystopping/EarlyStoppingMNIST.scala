package org.deeplearning4j.examples.misc.earlystopping

import java.util.concurrent.TimeUnit

import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.earlystopping.{EarlyStoppingConfiguration, EarlyStoppingModelSaver, EarlyStoppingResult}
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._

/**Early stopping example on a subset of MNIST
 * Idea: given a small subset of MNIST (1000 examples + 500 test set), conduct training and get the parameters that
 * have the minimum test set loss
 * This is an over-simplified example, but the principles used here should apply in more realistic cases.
 *
 * For further details on early stopping, see http://deeplearning4j.org/earlystopping.html
 *
 * @author Alex Black
 */
object EarlyStoppingMNIST {

    @throws[Exception]
    def main(args: Array[String]): Unit = {

        //Configure network:
        val nChannels = 1
        val outputNum = 10
        val batchSize = 25
        val iterations = 1
        val seed = 123
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.02)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(4)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20).dropOut(0.5)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false)
        new ConvolutionLayerSetup(builder,28,28,1)
        val configuration: MultiLayerConfiguration = builder.build()

        //Get data:
        val mnistTrain1024: DataSetIterator = new MnistDataSetIterator(batchSize,1024,false,true,true,12345)
        val mnistTest512: DataSetIterator = new MnistDataSetIterator(batchSize,512,false,false,true,12345)

        val tempDir: String = System.getProperty("java.io.tmpdir")
        val exampleDirectory: String = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/")

        val saver: EarlyStoppingModelSaver = new LocalFileModelSaver(exampleDirectory)
        val esConf: EarlyStoppingConfiguration = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(50)) //Max of 50 epochs
                .evaluateEveryNEpochs(1)
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES)) //Max of 20 minutes
                .scoreCalculator(new DataSetLossCalculator(mnistTest512, true))     //Calculate test set score
                .modelSaver(saver)
                .build()

        val trainer: EarlyStoppingTrainer = new EarlyStoppingTrainer(esConf,configuration,mnistTrain1024)

        //Conduct early stopping training:
        val result: EarlyStoppingResult = trainer.fit()
        System.out.println("Termination reason: " + result.getTerminationReason)
        System.out.println("Termination details: " + result.getTerminationDetails)
        System.out.println("Total epochs: " + result.getTotalEpochs)
        System.out.println("Best epoch number: " + result.getBestModelEpoch)
        System.out.println("Score at best epoch: " + result.getBestModelScore)

        //Print score vs. epoch
        val scoreVsEpoch = result.getScoreVsEpoch.asScala.map { case (k, v) => (k.toInt, v) }
        val list = scoreVsEpoch.keys.toSeq.sorted
        System.out.println("Score vs. Epoch:")
        list.foreach { i =>
            System.out.println(i + "\t" + scoreVsEpoch.get(i))
        }
    }
}
