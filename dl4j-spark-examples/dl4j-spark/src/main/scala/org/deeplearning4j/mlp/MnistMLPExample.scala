package org.deeplearning4j.mlp

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * Train a simple/small MLP on MNIST data using Spark, then evaluate it on the test set in a distributed manner
  *
  * Note that the network being trained here is too small to make proper use of Spark - but it shows the configuration
  * and evaluation used for Spark training.
  *
  *
  * To run the example locally: Run the example as-is. The example is set up to use Spark local by default.
  * NOTE: Spark local should only be used for development/testing. For data parallel training on a single machine
  * (for example, multi-GPU systems) instead use ParallelWrapper (which is faster than using Spark for training on a single machine).
  * See for example MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
  *
  * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
  * OR first modify the example by setting the field "useSparkLocal = false"
  *
  * @author Alex Black
  */
object MnistMLPExample {

  lazy val log = LoggerFactory.getLogger(classOf[MnistMLPExample])

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    new MnistMLPExample().entryPoint(args)
  }

}

class MnistMLPExample {

  @Parameter(names = Array("-useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
  private val useSparkLocal: Boolean = true

  @Parameter(names = Array("-batchSizePerWorker"), description = "Number of examples to fit each worker with")
  private val batchSizePerWorker: Int = 16

  @Parameter(names = Array("-numEpochs"), description = "Number of epochs for training")
  private val numEpochs: Int = 15

  @throws[Exception]
  protected def entryPoint(args: Array[String]) {
    //Handle command line arguments
    val jcmdr = new JCommander(this, Array[String]():_*)
    try {
      jcmdr.parse(args:_*)
    } catch { case e: ParameterException =>
      //User provides invalid input -> print the usage info
      jcmdr.usage()
      try { Thread.sleep(500) } catch { case e2: Exception => () }
      throw e
    }

    val sparkConf = new SparkConf
    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
    }
    sparkConf.setAppName("DL4J Spark MLP Example")
    val sc = new JavaSparkContext(sparkConf)

    //Load the data into memory then parallelize
    //This isn't a good approach in general - but is simple to use for this example
    val iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345)
    val iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345)
    val trainDataList = mutable.ArrayBuffer.empty[DataSet]
    val testDataList = mutable.ArrayBuffer.empty[DataSet]
    while (iterTrain.hasNext) {
        trainDataList += iterTrain.next
    }

    while (iterTest.hasNext) {
        testDataList += iterTest.next
    }

    val trainData = sc.parallelize(trainDataList)
    val testData = sc.parallelize(testDataList)


    //----------------------------------
    //Create network configuration and conduct network training
    val conf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .activation(Activation.LEAKYRELU)
      .weightInit(WeightInit.XAVIER)
      .learningRate(0.02)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .regularization(true)
      .l2(1e-4)
      .list
      .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build)
      .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build)
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX).nIn(100).nOut(10).build)
      .pretrain(false).backprop(true)
      .build

    //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker) //Each DataSet object: contains (by default) 32 examples
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)      //Async prefetching: 2 examples per worker
      .batchSizePerWorker(batchSizePerWorker)
      .build

    //Create the Spark network
    val sparkNet = new SparkDl4jMultiLayer(sc, conf, tm)

    //Execute training:
    var i: Int = 0
    for (i <- 0 until numEpochs) {
      sparkNet.fit(trainData)
      MnistMLPExample.log.info("Completed Epoch {}", i)
    }

    //Perform evaluation (distributed)
    val evaluation: Evaluation = sparkNet.evaluate(testData)
    MnistMLPExample.log.info("***** Evaluation *****")
    MnistMLPExample.log.info(evaluation.stats)

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

    MnistMLPExample.log.info("***** Example Complete *****")
  }
}
