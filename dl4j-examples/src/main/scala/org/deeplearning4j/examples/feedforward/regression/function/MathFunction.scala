package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray

trait MathFunction {
  def getFunctionValues(x: INDArray): INDArray

  def getName: String
}
