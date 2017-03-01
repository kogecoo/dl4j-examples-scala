package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * Calculate function value of Triangle Wave, which is an absolute value of Sawtooth Wave.
  *
  */
class TriangleWaveMathFunction extends MathFunction {
  def getFunctionValues(x: INDArray): INDArray = {
    val period = 6.0
    val xd = x.data.asDouble
    val yd = new Array[Double](xd.length)
    var i = 0
    for (i <- xd.indices) {
      yd(i) = Math.abs(2 * (xd(i) / period - Math.floor(xd(i) / period + 0.5)))
    }
    Nd4j.create(yd, Array[Int](xd.length, 1)) //Column vector
  }

  def getName: String = "TriangleWave"
}
