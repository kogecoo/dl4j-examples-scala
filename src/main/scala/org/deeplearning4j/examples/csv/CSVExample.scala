package org.deeplearning4j.examples.csv

import org.canova.api.records.reader.RecordReader
import org.canova.api.records.reader.impl.CSVRecordReader
import org.canova.api.split.FileSplit
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource

import java.util.Arrays
import scala.collection.JavaConverters._

object CSVExample {

    lazy val log = LoggerFactory.getLogger(CSVExample.getClass)

    def main(args: Array[String]) {
        val recordReader: RecordReader = new CSVRecordReader(0,",")
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()))
        //reader,label index,number of possible labels
        val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader,3,3)
        //get the dataset using the record reader. The datasetiterator handles vectorization
        val next: DataSet = iterator.next()
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10

        val numInputs = 4
        val outputNum = 3
        val iterations = 100
        val seed = 6L
        val listenerFreq = iterations/5


        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)

                .learningRate(1e-3)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build()

        //run the model
        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

        //split test and train
        val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.8)
        model.fit(testAndTrain.getTrain())

        //evaluate the model
        val eval = new Evaluation()
        val test: DataSet = testAndTrain.getTest()
        val output: INDArray = model.output(test.getFeatureMatrix())
        eval.eval(test.getLabels(), output)
        log.info(eval.stats())

    }

}
