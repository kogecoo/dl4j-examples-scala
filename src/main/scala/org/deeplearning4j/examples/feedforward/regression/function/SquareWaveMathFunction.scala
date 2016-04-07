package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Sign
import org.nd4j.linalg.api.ops.impl.transforms.Sin
import org.nd4j.linalg.factory.Nd4j


class SquareWaveMathFunction extends MathFunction {

    override def getFunctionValues(x: INDArray): INDArray = {
        val sin = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()))
        Nd4j.getExecutioner().execAndReturn(new Sign(sin))
    }

    override def getName(): String = "SquareWave"
}
