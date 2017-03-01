package org.deeplearning4j.examples.convolution

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

/**
  * Created by agibsonccc on 9/16/15.
  */
object LenetMnistExample {
  private val log = LoggerFactory.getLogger(LenetMnistExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    val nChannels = 1  // Number of input channels
    val outputNum = 10 // The number of possible outcomes
    val batchSize = 64 // Test batch size
    val nEpochs = 1    // Number of training epochs
    val iterations = 1 // Number of training iterations
    val seed = 123     //
    /*
       Create an iterator using the batch size for one iteration
     */
    log.info("Load data....")
    val mnistTrain = new MnistDataSetIterator(batchSize, true, 12345)
    val mnistTest = new MnistDataSetIterator(batchSize, false, 12345)
    /*
        Construct the neural network
     */ log.info("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true) // Training iterations as above
      .l2(0.0005)
      /*
        Uncomment the following for learning decay and bias
      */
      .learningRate(.01)//.biasLearningRate(0.02)
      //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.IDENTITY)
        .build)
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(2, new ConvolutionLayer.Builder(5, 5)
         //Note that nIn need not be specified in later layers
        .stride(1, 1)
        .nOut(50)
        .activation(Activation.IDENTITY)
        .build)
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(500)
        .build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX).build)
      .setInputType(InputType.convolutionalFlat(28, 28, 1)) //See note below
      .backprop(true).pretrain(false).build
    /*
    Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
    (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
    and the dense layer
    (b) Does some additional configuration validation
    (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
    layer based on the size of the previous layer (but it won't override values manually set by the user)

    InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
    For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
    MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
    row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
    */

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()

    log.info("Train model....")
    model.setListeners(new ScoreIterationListener(1))
    for (i <- 0 until nEpochs) {
      model.fit(mnistTrain)
      log.info("*** Completed epoch {} ***", i)
      log.info("Evaluate model....")
      val eval: Evaluation = new Evaluation(outputNum)
      while (mnistTest.hasNext) {
        {
          val ds: DataSet = mnistTest.next
          val output: INDArray = model.output(ds.getFeatureMatrix, false)
          eval.eval(ds.getLabels, output)
        }
      }
      log.info(eval.stats)
      mnistTest.reset()
    }
    log.info("****************Example finished********************")
  }
}
