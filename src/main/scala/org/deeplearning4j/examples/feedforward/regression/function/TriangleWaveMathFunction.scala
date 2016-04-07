package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class TriangleWaveMathFunction extends MathFunction {

    override def getFunctionValues(x: INDArray): INDArray = {
        val period = 6.0
        val xd: Array[Double] = x.data().asDouble()
        val yd: Array[Double] = xd.map { x => Math.abs(2 * (x / period - Math.floor(x / period + 0.5))) }
        Nd4j.create(yd, Array[Int](xd.length, 1))  //Column vector
    }

    override def getName(): String = "TriangleWave"
}
