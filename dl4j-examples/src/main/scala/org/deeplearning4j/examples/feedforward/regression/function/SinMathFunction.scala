package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Sin
import org.nd4j.linalg.factory.Nd4j

/**
  * Calculate function value of sine of x.
  */
class SinMathFunction extends MathFunction {
  def getFunctionValues(x: INDArray): INDArray = {
    Nd4j.getExecutioner.execAndReturn(new Sin(x.dup))
  }

  def getName: String = "Sin"
}
