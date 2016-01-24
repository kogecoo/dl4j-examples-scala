package org.deeplearning4j.examples.csv

import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
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

import scala.collection.JavaConverters._

object CSVExample {

    lazy val log = LoggerFactory.getLogger(CSVExample.getClass)

    def main(args: Array[String]) {
        val recordReader: RecordReader = new CSVRecordReader(0,",")
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()))
        //reader,label index,number of possible labels
        val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader,4,3)
        //get the dataset using the record reader. The datasetiterator handles vectorization
        val next: DataSet = iterator.next()
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10

        val numInputs = 4
        val outputNum = 3
        val iterations = 3
        val seed = 6L
        val listenerFreq = iterations


        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)

                .constrainGradientToUnitNorm(true).useDropConnect(true)
                .learningRate(1e-3)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("relu").dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build()

        //run the model
        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        next.normalizeZeroMeanZeroUnitVariance()
        next.shuffle()
        //split test and train
        val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.6)
        model.fit(testAndTrain.getTrain())

        //evaluate the model
        val eval = new Evaluation(3)
        val test: DataSet = testAndTrain.getTest()
        val output: INDArray = model.output(test.getFeatureMatrix())
        eval.eval(test.getLabels(), output)
        log.info(eval.stats())

    }

}
