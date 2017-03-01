package org.deeplearning4j.examples.dataExamples

import java.io.IOException

import org.apache.commons.io.IOUtils
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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, NormalizerStandardize}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.mutable

/**
  * This example is intended to be a simple CSV classifier that seperates the training data
  * from the test data for the classification of animals. It would be suitable as a beginner's
  * example because not only does it load CSV data into the network, it also shows how to extract the
  * data and display the results of the classification, as well as a simple method to map the lables
  * from the testing data into the results.
  *
  * @author Clay Graham
  */
object BasicCSVClassifier {
  private val log: Logger = LoggerFactory.getLogger(BasicCSVClassifier.getClass)
  private val eats: Map[Integer, String] = readEnumCSV("/DataExamples/animals/eats.csv")
  private val sounds: Map[Integer, String] = readEnumCSV("/DataExamples/animals/sounds.csv")
  private val classifiers: Map[Integer, String] = readEnumCSV("/DataExamples/animals/classifiers.csv")

  def main(args: Array[String]) {
    try {
      //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
      val labelIndex = 4 //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
      val numClasses = 3 //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2

      val batchSizeTraining = 30 //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
      val trainingData: DataSet = readCSVDataset("/DataExamples/animals/animals_train.csv", batchSizeTraining, labelIndex, numClasses)

      // this is the data we want to classify
      val batchSizeTest = 44
      val testData: DataSet = readCSVDataset("/DataExamples/animals/animals.csv", batchSizeTest, labelIndex, numClasses)


      // make the data model for records prior to normalization, because it
      // changes the data.
      val animals: Map[Integer, Map[String, AnyRef]] = makeAnimalsForTesting(testData)


      //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
      val normalizer: DataNormalization = new NormalizerStandardize
      normalizer.fit(trainingData) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
      normalizer.transform(trainingData) //Apply normalization to the training data
      normalizer.transform(testData) //Apply normalization to the test data. This is using statistics calculated from the *training* set

      val numInputs = 4
      val outputNum = 3
      val iterations = 1000
      val seed: Long = 6

      log.info("Build model....")
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
        .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build)
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
          .activation(Activation.SOFTMAX).nIn(3).nOut(outputNum).build)
        .backprop(true).pretrain(false)
        .build

      //run the model
      val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
      model.init()
      model.setListeners(new ScoreIterationListener(100))

      model.fit(trainingData)

      //evaluate the model on the test set
      val eval: Evaluation = new Evaluation(3)
      val output: INDArray = model.output(testData.getFeatureMatrix)

      eval.eval(testData.getLabels, output)
      log.info(eval.stats)

      val result = setFittedClassifiers(output, animals)
      logAnimals(result)
    } catch { case e: Exception => e.printStackTrace() }
  }

  def logAnimals(animals: Map[Integer, Map[String, Object]]) {
    for (a <- animals.values)
      log.info(a.toString)
  }

  def setFittedClassifiers(output: INDArray, animals: Map[Integer, Map[String, Object]]): Map[Integer, Map[String, Object]] = {
    animals.map { case (i, m) =>
      if ((0 until output.rows).contains(i)) {
        (i, m ++ Map("classifier" -> classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i))))))
      } else {
        (i, m)
      }
    }
  }

  /**
    * This method is to show how to convert the INDArray to a float array. This is to
    * provide some more examples on how to convert INDArray to types that are more java
    * centric.
    *
    * @param rowSlice
    * @return
    */
  def getFloatArrayFromSlice(rowSlice: INDArray): Array[Float] = {
    val result = new Array[Float](rowSlice.columns)
    for (i <- 0 until rowSlice.columns) {
      result(i) = rowSlice.getFloat(i)
    }
    result
  }

  /**
    * find the maximum item index. This is used when the data is fitted and we
    * want to determine which class to assign the test row to
    *
    * @param vals
    * @return
    */
  def maxIndex(vals: Array[Float]): Int = {
    var maxIndex = 0
    for (i <- 1 until vals.length) {
      val newnumber: Float = vals(i)
      if (newnumber > vals(maxIndex)) {
        maxIndex = i
      }
    }
    maxIndex
  }

  /**
    * take the dataset loaded for the matric and make the record model out of it so
    * we can correlate the fitted classifier to the record.
    *
    * @param testData
    * @return
    */
  def makeAnimalsForTesting(testData: DataSet): Map[Integer, Map[String, Object]] = {
    val animals = mutable.Map.empty[Integer, Map[String, Object]]
    val features: INDArray = testData.getFeatureMatrix
    for (i <- 0 until features.rows) {
      val slice: INDArray = features.slice(i)
      val animal = mutable.Map.empty[String, Object]
      //set the attributes
      animal.update("yearsLived", slice.getInt(0).asInstanceOf[Object])
      animal.update("eats", eats.get(slice.getInt(1)))
      animal.update("sounds", sounds.get(slice.getInt(2)))
      animal.update("weight", slice.getFloat(3).asInstanceOf[Object])
      animals.update(i, animal.toMap)
    }
    animals.toMap
  }

  def readEnumCSV(csvFileClasspath: String): Map[Integer, String] = {
    try {
      val lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream)
      val enums = mutable.Map.empty[Integer, String]
      for (line <- lines) {
        val parts = line.split(",")
        enums.update(parts(0).toInt, parts(1))
      }
      enums.toMap
    } catch { case e: Exception =>
        e.printStackTrace()
        null
    }
  }

  /**
    * used for testing and training
    *
    * @param csvFileClasspath
    * @param batchSize
    * @param labelIndex
    * @param numClasses
    * @return
    * @throws IOException
    * @throws InterruptedException
    */
  @throws[IOException]
  @throws[InterruptedException]
  private def readCSVDataset(csvFileClasspath: String, batchSize: Int, labelIndex: Int, numClasses: Int): DataSet = {
    val rr = new CSVRecordReader
    rr.initialize(new FileSplit(new ClassPathResource(csvFileClasspath).getFile))
    val iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses)
    iterator.next
  }
}
