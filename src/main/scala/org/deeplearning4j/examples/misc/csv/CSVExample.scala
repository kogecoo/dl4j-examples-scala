package org.deeplearning4j.examples.misc.csv

import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource

import java.util.Collections


object CSVExample {

    lazy val log = LoggerFactory.getLogger(CSVExample.getClass)

    def main(args: Array[String]) {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        val numLinesToSkip = 0
        val delimiter = ","
        val recordReader: RecordReader = new CSVRecordReader(numLinesToSkip,delimiter)
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile))

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        val labelIndex = 4
        val numClasses = 3
        val batchSize = 150    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);


        val next: DataSet = iterator.next()

        val numInputs = 4
        val outputNum = 3
        val iterations = 1000
        val seed = 6L
        val listenerFreq = iterations


        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)

                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(3).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build()

        //run the model
        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new ScoreIterationListener(100))

        //Normalize the full data set. Our DataSet 'next' contains the full 150 examples
        next.normalizeZeroMeanZeroUnitVariance()
        next.shuffle()
        //split test and train
        val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.65)

        val trainingData = testAndTrain.getTrain()
        model.fit(trainingData)

        //evaluate the model on the test set
        val eval = new Evaluation(3)
        val test: DataSet = testAndTrain.getTest
        val output: INDArray = model.output(test.getFeatureMatrix)
        eval.eval(test.getLabels, output)
        log.info(eval.stats())
    }

}
