package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Sin
import org.nd4j.linalg.factory.Nd4j


class SinMathFunction extends MathFunction {

    override def getFunctionValues(x: INDArray): INDArray = {
        Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()))
    }

    override def getName(): String = "Sin"
}
