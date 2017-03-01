package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.{Sign, Sin}
import org.nd4j.linalg.factory.Nd4j

/**
  * Sign(x) or Sign of a real number, x, is -1 if x is negative, 0 if x is zero and 1 if x is positive.
  *
  * Calculate function value of Sign of Sine of x, which can be -1, 0 or 1.
  * The three possible outputs of Sign(sin) will form a line that resembles "squares" in the graph.
  */
class SquareWaveMathFunction extends MathFunction {
  def getFunctionValues(x: INDArray): INDArray = {
    val sin = Nd4j.getExecutioner.execAndReturn(new Sin(x.dup))
    Nd4j.getExecutioner.execAndReturn(new Sign(sin))
  }

  def getName: String = "SquareWave"
}
