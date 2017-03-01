package org.deeplearning4j.examples.misc.lossfunctions

import lombok.EqualsAndHashCode
import org.apache.commons.math3.util.Pair
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.{Logger, LoggerFactory}


@EqualsAndHashCode
object CustomLossL1L2 {

  private val logger: Logger = LoggerFactory.getLogger(classOf[CustomLossL1L2])

}

@EqualsAndHashCode
class CustomLossL1L2 extends ILossFunction {
  /*
   Needs modification depending on your loss function
       scoreArray calculates the loss for a single data point or in other words a batch size of one
       It returns an array the shape and size of the output of the neural net.
       Each element in the array is the loss function applied to the prediction and it's true value
       scoreArray takes in:
       true labels - labels
       the input to the final/output layer of the neural network - preOutput,
       the activation function on the final layer of the neural network - activationFn
       the mask - (if there is a) mask associated with the label
    */
  private def scoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray): INDArray = {
    var scoreArr: INDArray = null
    // This is the output of the neural network, the y_hat in the notation above
    //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
    val output: INDArray = activationFn.getActivation(preOutput.dup, true)
    //The score is calculated as the sum of (y-y_hat)^2 + |y - y_hat|
    val yMinusyHat: INDArray = Transforms.abs(labels.sub(output))
    scoreArr = yMinusyHat.mul(yMinusyHat)
    scoreArr.addi(yMinusyHat)
    if (mask != null) {
      scoreArr.muliColumnVector(mask)
    }
    scoreArr
  }

  /*
  Remains the same for all loss functions
  Compute Score computes the average loss function across many datapoints.
  The loss for a single datapoint is summed over all output features.
   */
  def computeScore(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray, average: Boolean): Double = {
    val scoreArr = scoreArray(labels, preOutput, activationFn, mask)

    var score = scoreArr.sumNumber.doubleValue

    if (average) {
      score /= scoreArr.size(0)
    }
    score
  }

  /*
  Remains the same for all loss functions
  Compute Score computes the loss function for many datapoints.
  The loss for a single datapoint is the loss summed over all output features.
  Returns an array that is #of samples x size of the output feature
   */
  def computeScoreArray(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray): INDArray = {
    val scoreArr: INDArray = scoreArray(labels, preOutput, activationFn, mask)
    scoreArr.sum(1)
  }

  /*
  Needs modification depending on your loss function
      Compute the gradient wrt to the preout (which is the input to the final layer of the neural net)
      Use the chain rule
      In this case L = (y - yhat)^2 + |y - yhat|
      dL/dyhat = -2*(y-yhat) - sign(y-yhat), sign of y - yhat = +1 if y-yhat>= 0 else -1
      dyhat/dpreout = d(Activation(preout))/dpreout = Activation'(preout)
      dL/dpreout = dL/dyhat * dyhat/dpreout
  */
  def computeGradient(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray): INDArray = {
    val output: INDArray = activationFn.getActivation(preOutput.dup, true)
    /*
    //NOTE: There are many ways to do this same set of operations in nd4j
    //The following is the most readable for the sake of this example, not necessarily the fastest
    //Refer to the Implementation of LossL1 and LossL2 for more efficient ways
     */
    val yMinusyHat: INDArray = labels.sub(output)
    val dldyhat: INDArray = yMinusyHat.mul(-2).sub(Transforms.sign(yMinusyHat))

    //d(L)/d(yhat) -> this is the line that will change with your loss function
    //Everything below remains the same
    val dLdPreOut: INDArray = activationFn.backprop(preOutput.dup, dldyhat).getFirst
    //multiply with masks, always
    if (mask != null) {
      dLdPreOut.muliColumnVector(mask)
    }
    dLdPreOut
  }

  //remains the same for a custom loss function
  override def computeGradientAndScore(labels: INDArray, preOutput: INDArray, activationFn: IActivation, mask: INDArray, average: Boolean): Pair[java.lang.Double, INDArray] = {
    new Pair[java.lang.Double, INDArray](
      computeScore(labels, preOutput, activationFn, mask, average),
      computeGradient(labels, preOutput, activationFn, mask)
    )
  }

  override def toString: String = "CustomLossL1L2()"

}
