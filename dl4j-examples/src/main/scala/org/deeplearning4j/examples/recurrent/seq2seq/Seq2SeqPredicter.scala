package org.deeplearning4j.examples.recurrent.seq2seq

import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

/**
  * Created by susaneraly on 1/11/17.
  */
/*
    Note this is a helper class with methods to step through the decoder, one time step at a time.
    This process is common to all seq2seq models and will eventually be wrapped in a class in dl4j (along with an easier API).
    Track issue:
        https://github.com/deeplearning4j/deeplearning4j/issues/2635
 */
class Seq2SeqPredicter(var net: ComputationGraph) {
  private var decoderInputTemplate: INDArray = null

  def output(testSet: MultiDataSet): INDArray = {
    if (testSet.getFeatures(0).size(0) > 2) {
      output(testSet, print = false)
    } else {
      output(testSet, print = true)
    }
  }

  def output(testSet: MultiDataSet, print: Boolean): INDArray = {
    val correctOutput = testSet.getLabels(0)
    var ret = Nd4j.zeros(correctOutput.shape: _*)
    decoderInputTemplate = testSet.getFeatures(1).dup

    var currentStepThrough = 0
    val stepThroughs = correctOutput.size(2) - 1
    while (currentStepThrough < stepThroughs) {
      if (print) {
        println("In time step " + currentStepThrough)
        println("\tEncoder input and Decoder input:")
        println(CustomSequenceIterator.mapToString(testSet.getFeatures(0), decoderInputTemplate, " +  "))
      }
      ret = stepOnce(testSet, currentStepThrough)
      if (print) {
        println("\tDecoder output:")
        println("\t" + CustomSequenceIterator.oneHotDecode(ret).mkString("\n\t"))
      }
      currentStepThrough += 1
    }
    ret = net.output(false, testSet.getFeatures(0), decoderInputTemplate)(0)
    if (print) {
      println("Final time step " + currentStepThrough)
      println("\tEncoder input and Decoder input:")
      println(CustomSequenceIterator.mapToString(testSet.getFeatures(0), decoderInputTemplate, " +  "))
      println("\tDecoder output:")
      println("\t" + CustomSequenceIterator.oneHotDecode(ret).mkString("\n\t"))
    }
    ret
  }

  /*
      Will do a forward pass through encoder + decoder with the given input
      Updates the decoder input template from time = 1 to time t=n+1;
      Returns the output from this forward pass
   */
  private def stepOnce(testSet: MultiDataSet, n: Int): INDArray = {
    val currentOutput: INDArray = net.output(false, testSet.getFeatures(0), decoderInputTemplate)(0)
    copyTimeSteps(n, currentOutput, decoderInputTemplate)
    currentOutput
  }

  /*
      Copies timesteps
      time = 0 to time = t in "fromArr"
      to time = 1 to time = t+1 in "toArr"
   */
  private def copyTimeSteps(t: Int, fromArr: INDArray, toArr: INDArray) {
    val fromView: INDArray = fromArr.get(NDArrayIndex.all, NDArrayIndex.all, NDArrayIndex.interval(0, t, true))
    val toView: INDArray = toArr.get(NDArrayIndex.all, NDArrayIndex.all, NDArrayIndex.interval(1, t + 1, true))
    toView.assign(fromView.dup)
  }

}
