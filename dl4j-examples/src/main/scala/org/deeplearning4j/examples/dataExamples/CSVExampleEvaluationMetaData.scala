package org.deeplearning4j.examples.dataExamples

import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * This example is a version of the basic CSV example, but adds the following:
  * (a) Meta data tracking - i.e., where data for each example comes from
  * (b) Additional evaluation information - getting metadata for prediction errors
  *
  * @author Alex Black
  */
object CSVExampleEvaluationMetaData {
  @throws[Exception]
  def main(args: Array[String]) {
    //First: get the dataset using the record reader. This is as per CSV example - see that example for details
    val recordReader = new CSVRecordReader(0, ",")
    recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile))
    val labelIndex = 4
    val numClasses = 3
    val batchSize = 150

    val iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)
    iterator.setCollectMetaData(true) //Instruct the iterator to collect metadata, and store it in the DataSet objects
    val allData = iterator.next
    allData.shuffle(123)
    val testAndTrain = allData.splitTestAndTrain(0.65) //Use 65% of data for training

    val trainingData = testAndTrain.getTrain
    val testData = testAndTrain.getTest

    //Let's view the example metadata in the training and test sets:
    val trainMetaData = trainingData.getExampleMetaData(classOf[RecordMetaData])
    val testMetaData = testData.getExampleMetaData(classOf[RecordMetaData])

    //Let's show specifically which examples are in the training and test sets, using the collected metadata
    println("  +++++ Training Set Examples MetaData +++++")
    val format: String = "%-20s\t%s"
    for (recordMetaData <- trainMetaData.asScala) {
      println(String.format(format, recordMetaData.getLocation, recordMetaData.getURI))
      //Also available: recordMetaData.getReaderClass()
    }
    println("\n\n  +++++ Test Set Examples MetaData +++++")
    for (recordMetaData <- testMetaData.asScala) {
      println(recordMetaData.getLocation)
    }


    //Normalize data as per basic CSV example
    val normalizer = new NormalizerStandardize
    normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
    normalizer.transform(trainingData) //Apply normalization to the training data
    normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set


    //Configure a simple model. We're not using an optimal configuration here, in order to show evaluation/errors, later
    val numInputs = 4
    val outputNum = 3
    val iterations = 50
    val seed = 6

    println("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(Activation.TANH)
      .weightInit(WeightInit.XAVIER)
      .learningRate(0.1)
      .regularization(true)
      .l2(1e-4)
      .list
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3).build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build)
      .backprop(true).pretrain(false)
      .build

    //Fit the model
    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    model.fit(trainingData)

    //Evaluate the model on the test set
    val eval = new Evaluation(3)
    val output = model.output(testData.getFeatureMatrix)
    eval.eval(testData.getLabels, output, testMetaData) //Note we are passing in the test set metadata here
    println(eval.stats)

    //Get a list of prediction errors, from the Evaluation object
    //Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
    val predictionErrors = eval.getPredictionErrors
    println("\n\n+++++ Prediction Errors +++++")
    for (p <- predictionErrors.asScala) {
      println("Predicted class: " + p.getPredictedClass + ", Actual class: " + p.getActualClass + "\t" + p.getRecordMetaData(classOf[RecordMetaData]).getLocation)
    }
    //We can also load a subset of the data, to a DataSet object:
    val predictionErrorMetaData = mutable.ArrayBuffer.empty[RecordMetaData]
    for (p <- predictionErrors.asScala) predictionErrorMetaData += p.getRecordMetaData(classOf[RecordMetaData])
    val predictionErrorExamples: DataSet = iterator.loadFromMetaData(predictionErrorMetaData.asJava)
    normalizer.transform(predictionErrorExamples) //Apply normalization to this subset

    //We can also load the raw data:
    val predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData.asJava)

    //Print out the prediction errors, along with the raw data, normalized data, labels and network predictions:
    for (i <- predictionErrors.asScala.indices) {
      val p = predictionErrors.get(i)
      val meta = p.getRecordMetaData(classOf[RecordMetaData])
      val features = predictionErrorExamples.getFeatures.getRow(i)
      val labels = predictionErrorExamples.getLabels.getRow(i)
      val rawData  = predictionErrorRawData.get(i).getRecord
      val networkPrediction = model.output(features)
      println(meta.getLocation + ": " + "\tRaw Data: " + rawData + "\tNormalized: " + features + "\tLabels: " + labels + "\tPredictions: " + networkPrediction)
    }

    //Some other useful evaluation methods:
    val list1 = eval.getPredictions(1, 2) //Predictions: actual class 1, predicted class 2
    val list2 = eval.getPredictionByPredictedClass(2) //All predictions for predicted class 2
    val list3 = eval.getPredictionsByActualClass(2) //All predictions for actual class 2
  }

}
