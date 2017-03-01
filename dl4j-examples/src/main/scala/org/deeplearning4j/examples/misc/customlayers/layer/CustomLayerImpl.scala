package org.deeplearning4j.examples.misc.customlayers.layer

import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.gradient.{DefaultGradient, Gradient}
import org.deeplearning4j.nn.layers.BaseLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

/**
  * Layer (implementation) class for the custom layer example
  *
  * @author Alex Black
  */
class CustomLayerImpl(conf: NeuralNetConfiguration) extends BaseLayer[CustomLayer](conf) {

  override def preOutput(x: INDArray, training: Boolean): INDArray = {
    /*
    The preOut method(s) calculate the activations (forward pass), before the activation function is applied.

    Because we aren't doing anything different to a standard dense layer, we can use the existing implementation
    for this. Other network types (RNNs, CNNs etc) will require you to implement this method.

    For custom layers, you may also have to implement methods such as calcL1, calcL2, numParams, etc.
    */
    super.preOutput(x, training)
  }

  override def activate(training: Boolean): INDArray = {
    /*
    The activate method is used for doing forward pass. Note that it relies on the pre-output method;
    essentially we are just applying the activation function (or, functions in this example).
    In this particular (contrived) example, we have TWO activation functions - one for the first half of the outputs
    and another for the second half.
     */

    val output = preOutput(training)
    val columns = output.columns

    val firstHalf = output.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val secondHalf = output.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val activation1 = conf.getLayer.getActivationFn
    val activation2 = conf.getLayer.asInstanceOf[CustomLayer].getSecondActivationFunction

    //IActivation function instances modify the activation functions in-place
    activation1.getActivation(firstHalf, training)
    activation2.getActivation(secondHalf, training)

    output
  }

  def isPretrainLayer: Boolean = { false }

  override def backpropGradient(epsilon: INDArray): Pair[Gradient, INDArray] = {
    /*
    The baockprop gradient method here is very similar to the BaseLayer backprop gradient implementation
    The only major difference is the two activation functions we have added in this example.

    Note that epsilon is dL/da - i.e., the derivative of the loss function with respect to the activations.
    It has the exact same shape as the activation arrays (i.e., the output of preOut and activate methods)
    This is NOT the 'delta' commonly used in the neural network literature; the delta is obtained from the
    epsilon ("epsilon" is dl4j's notation) by doing an element-wise product with the activation function derivative.

    Note the following:
    1. Is it very important that you use the gradientViews arrays for the results.
       Note the gradientViews.get(...) and the in-place operations here.
       This is because DL4J uses a single large array for the gradients for efficiency. Subsets of this array (views)
       are distributed to each of the layers for efficient backprop and memory management.
    2. The method returns two things, as a Pair:
       (a) a Gradient object (essentially a Map<String,INDArray> of the gradients for each parameter (again, these
           are views of the full network gradient array)
       (b) an INDArray. This INDArray is the 'epsilon' to pass to the layer below. i.e., it is the gradient with
           respect to the input to this layer
    */
    val activationDerivative = preOutput(true)
    val columns = activationDerivative.columns

    val firstHalf = activationDerivative.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val secondHalf = activationDerivative.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val epsilonFirstHalf = epsilon.get(NDArrayIndex.all, NDArrayIndex.interval(0, columns / 2))
    val epsilonSecondHalf = epsilon.get(NDArrayIndex.all, NDArrayIndex.interval(columns / 2, columns))

    val activation1 = conf.getLayer.getActivationFn
    val activation2 = (conf.getLayer.asInstanceOf[CustomLayer]).getSecondActivationFunction

    //IActivation backprop method modifies the 'firstHalf' and 'secondHalf' arrays in-place, to contain dL/dz
    activation1.backprop(firstHalf, epsilonFirstHalf)
    activation2.backprop(secondHalf, epsilonSecondHalf)

    //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
    //        INDArray delta = epsilon.muli(activationDerivative);

    if (maskArray != null) {
      activationDerivative.muliColumnVector(maskArray)
    }

    val ret: Gradient = new DefaultGradient

    val weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY) //f order
    Nd4j.gemm(input, activationDerivative, weightGrad, true, false, 1.0, 0.0)
    val biasGrad: INDArray = gradientViews.get(DefaultParamInitializer.BIAS_KEY)
    biasGrad.assign(activationDerivative.sum(0)) //TODO: do this without the assign

    ret.gradientForVariable.put(DefaultParamInitializer.WEIGHT_KEY, weightGrad)
    ret.gradientForVariable.put(DefaultParamInitializer.BIAS_KEY, biasGrad)

    val epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose).transpose
    new Pair[Gradient, INDArray](ret, epsilonNext)
  }
}
