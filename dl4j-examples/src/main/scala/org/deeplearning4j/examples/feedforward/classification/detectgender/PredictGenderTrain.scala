package org.deeplearning4j.examples.feedforward.classification.detectgender

/**
  * Created by KIT Solutions (www.kitsol.com) on 9/28/2016.
  */

import java.io._

import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * - Notes:
  *  - Data files are stored at following location
  * .\dl4j-0.4-examples-master\dl4j-examples\src\main\resources\PredictGender\Data folder
  */
object PredictGenderTrain {
  def main(args: Array[String]) {
    val dg: PredictGenderTrain = new PredictGenderTrain
    dg.filePath = System.getProperty("user.dir") + "/dl4j-examples/src/main/resources/PredictGender/Data"
    dg.train()
  }
}

class PredictGenderTrain {
  var filePath: String = null

  /**
    * This function uses GenderRecordReader and passes it to RecordReaderDataSetIterator for further training.
    */
  def train() {
    val seed = 123456
    val learningRate = 0.005
    // was .01 but often got errors o.d.optimize.solvers.BaseOptimizer - Hit termination condition on iteration 0"
    val batchSize = 100
    val nEpochs = 100
    var numInputs = 0
    var numOutputs = 0
    var numHiddenNodes = 0

    try {
      val rr = new GenderRecordReader(List("M", "F"))
      try {
        val st = System.currentTimeMillis
        println("Preprocessing start time : " + st)
        rr.initialize(new FileSplit(new File(this.filePath)))
        val et: Long = System.currentTimeMillis
        println("Preprocessing end time : " + et)
        println("time taken to process data : " + (et - st) + " ms")
        numInputs = rr.maxLengthName * 5 // multiplied by 5 as for each letter we use five binary digits like 00000
        numOutputs = 2
        numHiddenNodes = 2 * numInputs + numOutputs
        val rr1 = new GenderRecordReader(List("M", "F"))
        rr1.initialize(new FileSplit(new File(this.filePath)))
        val trainIter = new RecordReaderDataSetIterator(rr, batchSize, numInputs, 2)
        val testIter = new RecordReaderDataSetIterator(rr1, batchSize, numInputs, 2)
        val conf = new NeuralNetConfiguration.Builder()
          .seed(seed)
          .biasInit(1)
          .regularization(true)
          .l2(1e-4)
          .iterations(1)
          .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          .learningRate(learningRate)
          .updater(Updater.NESTEROVS)
          .momentum(0.9)
          .list
          .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .build())
          .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.RELU)
            .build())
          .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build())
          .pretrain(false).backprop(true).build

        val model = new MultiLayerNetwork(conf)
        model.init()
        model.setListeners(new HistogramIterationListener(10))
        for (n <- 0 until nEpochs) {
          while (trainIter.hasNext) {
            model.fit(trainIter.next())
          }
          trainIter.reset()
        }
        ModelSerializer.writeModel(model, this.filePath + "/PredictGender.net", true)

        println("Evaluate model....")
        val eval = new Evaluation(numOutputs)
        while (testIter.hasNext) {
          val t = testIter.next()
          val features = t.getFeatureMatrix
          val lables = t.getLabels
          val predicted = model.output(features, false)
          eval.eval(lables, predicted)
        }
        //Print the evaluation statistics
        println(eval.stats)
      } catch { case e: Exception =>
        println("Exception111 : " + e.getMessage)
        println(e.getStackTraceString)
      } finally {
        if (rr != null) rr.close()
      }
    }
  }
}
