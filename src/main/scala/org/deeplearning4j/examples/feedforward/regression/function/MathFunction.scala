package org.deeplearning4j.examples.feedforward.regression.function

import org.nd4j.linalg.api.ndarray.INDArray

abstract class MathFunction {

    def getFunctionValues(x: INDArray): INDArray

    def getName(): String
}
