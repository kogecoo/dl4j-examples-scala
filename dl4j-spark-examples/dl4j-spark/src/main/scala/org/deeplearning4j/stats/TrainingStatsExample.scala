package org.deeplearning4j.stats

import java.io.File

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.rnn.SparkLSTMCharacterExample
import org.deeplearning4j.spark.api.stats.SparkTrainingStats
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.spark.stats.{EventStats, StatsUtils}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/**
  * This example is designed to show how to use DL4J's Spark training benchmarking/debugging/timing functionality.
  * For details: See https://deeplearning4j.org/spark#sparkstats
  *
  * The idea with this tool is to capture statistics on various aspects of Spark training, in order to identify
  * and debug performance issues.
  *
  * For the sake of the example, we will be using a network configuration and data as per the SparkLSTMCharacterExample.
  *
  *
  * To run the example locally: Run the example as-is. The example is set up to use Spark local.
  *
  * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
  * OR first modify the example by setting the field "useSparkLocal = false"
  *
  * NOTE: On some clusters without internet access, this example may fail with "Error querying NTP server"
  * See: https://deeplearning4j.org/spark#sparkstatsntp
  *
  * @author Alex Black
  */
object TrainingStatsExample {

  val log: Logger = LoggerFactory.getLogger(classOf[TrainingStatsExample])

  @throws[Exception]
  def main(args: Array[String]) {
    new TrainingStatsExample().entryPoint(args)
  }

  //Configuration for the network we will be training
  private def getConfiguration: MultiLayerConfiguration = {
    val lstmLayerSize = 200 //Number of units in each GravesLSTM layer
    val tbpttLength = 50 //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters

    val CHAR_TO_INT = SparkLSTMCharacterExample.getCharToInt
    val nIn = CHAR_TO_INT.size
    val nOut = CHAR_TO_INT.size

    //Set up network configuration:
    new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .updater(Updater.RMSPROP)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(nOut).build()) //MCXENT + softmax for classification
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build()

  }
}

class TrainingStatsExample {

  @Parameter(names = Array("-useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1) private val useSparkLocal: Boolean = true
  @throws[Exception]
  private def entryPoint(args: Array[String]) {
    //Handle command line arguments
    val jcmdr = new JCommander(this, Array(): _*)
    try {
      jcmdr.parse(args:_*)
    } catch { case e: ParameterException =>
      //User provides invalid input -> print the usage info
      jcmdr.usage()
      try {
        Thread.sleep(500)
      } catch { case e2: Exception => () }
      throw e
    }


    //Set up network configuration:
    val config = TrainingStatsExample.getConfiguration

    //Set up the Spark-specific configuration
    val examplesPerWorker = 8   //i.e., minibatch size that each worker gets
    val averagingFrequency = 3  //Frequency with which parameters are averaged

    //Set up Spark configuration and context
    val sparkConf: SparkConf = new SparkConf
    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
      TrainingStatsExample.log.info("Using Spark Local")
    }
    sparkConf.setAppName("DL4J Spark Stats Example")
    val sc: JavaSparkContext = new JavaSparkContext(sparkConf)

    //Get data. See SparkLSTMCharacterExample for details
    val trainingData: JavaRDD[DataSet] = SparkLSTMCharacterExample.getTrainingData(sc)

    //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
    //Here, we are using standard parameter averaging
    val examplesPerDataSetObject = 1  //We haven't pre-batched our data: therefore each DataSet object contains 1 example
    val tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
      .workerPrefetchNumBatches(2) //Async prefetch 2 batches for each worker
      .averagingFrequency(averagingFrequency)
      .batchSizePerWorker(examplesPerWorker)
      .build

    //Create the Spark network
    val sparkNetwork: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, config, tm)

    //*** Tell the network to collect training statistics. These will NOT be collected by default ***
    sparkNetwork.setCollectTrainingStats(true)

    //Fit for 1 epoch:
    sparkNetwork.fit(trainingData)

    //Delete the temp training files, now that we are done with them (if fitting for multiple epochs: would be re-used)
    tm.deleteTempFiles(sc)

    //Get the statistics:
    val stats: SparkTrainingStats = sparkNetwork.getSparkTrainingStats
    val statsKeySet = stats.getKeySet //Keys for the types of statistics
    println("--- Collected Statistics ---")
    for (s <- statsKeySet.asScala) {
      println(s)
    }
    //Demo purposes: get one statistic and print it
    val first = statsKeySet.iterator.next
    val firstStatEvents = stats.getValue(first)
    val es: EventStats = firstStatEvents.get(0)
    TrainingStatsExample.log.info("Training stats example:")
    TrainingStatsExample.log.info("Machine ID:     " + es.getMachineID)
    TrainingStatsExample.log.info("JVM ID:         " + es.getJvmID)
    TrainingStatsExample.log.info("Thread ID:      " + es.getThreadID)
    TrainingStatsExample.log.info("Start time ms:  " + es.getStartTime)
    TrainingStatsExample.log.info("Duration ms:    " + es.getDurationMs)
    //Export a HTML file containing charts of the various stats calculated during training
    StatsUtils.exportStatsAsHtml(stats, "SparkStats.html", sc)
    TrainingStatsExample.log.info("Training stats exported to {}", new File("SparkStats.html").getAbsolutePath)
    TrainingStatsExample.log.info("****************Example finished********************")
  }
}
