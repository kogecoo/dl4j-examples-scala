package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class SawtoothMathFunction extends MathFunction {

    override def getFunctionValues(x: INDArray): INDArray = {
        val sawtoothPeriod = 4.0
        val xd2: Array[Double] = x.data().asDouble()
        val yd2: Array[Double] = xd2.map { x => 2 * (x / sawtoothPeriod - Math.floor(x / sawtoothPeriod + 0.5)) }
        Nd4j.create(yd2, Array[Int](xd2.length, 1))  //Column vector
    }

    override def getName(): String = "Sawtooth"
}
