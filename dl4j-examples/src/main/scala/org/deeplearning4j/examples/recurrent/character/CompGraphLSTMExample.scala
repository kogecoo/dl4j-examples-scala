package org.deeplearning4j.examples.recurrent.character

import java.util.Random

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{BackpropType, ComputationGraphConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * This example is almost identical to the GravesLSTMCharModellingExample, except that it utilizes the ComputationGraph
  * architecture instead of MultiLayerNetwork architecture. See the javadoc in that example for details.
  * For more details on the ComputationGraph architecture, see http://deeplearning4j.org/compgraph
  *
  * In addition to the use of the ComputationGraph a, this version has skip connections between the first and output layers,
  * in order to show how this configuration is done. In practice, this means we have the following types of connections:
  * (a) first layer -> second layer connections
  * (b) first layer -> output layer connections
  * (c) second layer -> output layer connections
  *
  * @author Alex Black
  */
object CompGraphLSTMExample {

  @throws[Exception]
  def main(args: Array[String]) {

    val lstmLayerSize = 200                        //Number of units in each GravesLSTM layer
    val miniBatchSize = 32                         //Size of mini batch to use when  training
    val exampleLength = 1000                       //Length of each training example sequence to use. This could certainly be increased
    val tbpttLength = 50                           //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val numEpochs = 1                              //Total number of training epochs
    val generateSamplesEveryNMinibatches = 10      //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    val nSamplesToGenerate = 4                     //Number of samples to generate after each training epoch
    val nCharactersToSample = 300                  //Length of each sample to generate
    val generationInitialization: String = null    //Optional character initialization; a random character is used if null
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    val rng = new Random(12345)

    //Get a DataSetIterator that handles vectorization of text into something we can use to train
    // our GravesLSTM network.
    val iter: CharacterIterator = GravesLSTMCharModellingExample.getShakespeareIterator(miniBatchSize, exampleLength)
    val nOut = iter.totalOutcomes

    //Set up network configuration:
    val conf: ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .graphBuilder.addInputs("input") //Give the input a name. For a ComputationGraph with multiple inputs, this also defines the input array orders
      //First layer: name "first", with inputs from the input called "input"
      .addLayer("first", new GravesLSTM.Builder().nIn(iter.inputColumns).nOut(lstmLayerSize)
        .updater(Updater.RMSPROP).activation(Activation.TANH).build, "input")
      //Second layer, name "second", with inputs from the layer called "first"
      .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
        .updater(Updater.RMSPROP).activation(Activation.TANH).build, "first")
      //Output layer, name "outputlayer" with inputs from the two layers called "first" and "second"
      .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX).updater(Updater.RMSPROP)
        .nIn(2 * lstmLayerSize).nOut(nOut).build, "first", "second")
      .setOutputs("outputLayer") //List the output. For a ComputationGraph with multiple outputs, this also defines the input array orders
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build

    val net = new ComputationGraph(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    //Print the  number of parameters in the network (and for each layer)
    var totalNumParams: Int = 0
    for (i <- 0 until net.getNumLayers) {
      val nParams = net.getLayer(i).numParams
      println("Number of parameters in layer " + i + ": " + nParams)
      totalNumParams += nParams
    }
    println("Total number of network parameters: " + totalNumParams)

    //Do training, and then generate and print samples from network
    var miniBatchNumber = 0
    for (i <- 0 until numEpochs) {
      while (iter.hasNext) {
        val ds: DataSet = iter.next
        net.fit(ds)
        miniBatchNumber += 1
        if (miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
          println("--------------------")
          println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters")
          println("Sampling characters from network given initialization \"" +
            (if (generationInitialization == null) "" else generationInitialization) + "\"")
          val samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate)
          for (j <- samples.indices) {
            println("----- Sample " + j + " -----")
            println(samples(j))
            println()
          }
        }
      }
      iter.reset() //Reset iterator for another epoch
    }
    println("\n\nExample complete")
  }

  /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
    * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
    * Note that the initalization is used for all samples
    *
    * @param _initialization    String, may be null. If null, select a random character as initialization for all samples
    * @param charactersToSample Number of characters to sample from network (excluding initialization)
    * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
    * @param iter               CharacterIterator. Used for going from indexes back to characters
    */
  private def sampleCharactersFromNetwork(_initialization: String, net: ComputationGraph,
                                          iter: CharacterIterator, rng: Random, charactersToSample: Int, numSamples: Int): Array[String] = {
    //Set up initialization. If no initialization: use a random character
    val initialization = if (_initialization == null) {
      String.valueOf(iter.getRandomCharacter)
    } else _initialization

    //Create input for initialization
    val initializationInput = Nd4j.zeros(Seq(numSamples, iter.inputColumns, initialization.length):_*)
    val init = initialization.toCharArray
    for (i <- init.indices) {
      val idx = iter.convertCharacterToIndex(init(i))
      for (j <- 0 until numSamples) {
        initializationInput.putScalar(Array[Int](j, idx, i), 1.0f)
      }
    }
    val sb = new Array[StringBuilder](numSamples)
    for (i <- 0 until numSamples) {
      sb(i) = new StringBuilder(initialization)
    }

    //Sample from network (and feed samples back into input) one character at a time (for all samples)
    //Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)(0)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    for (i <- 0 until charactersToSample) {
      //Set up next input (single time step) by sampling from previous output
      val nextInput: INDArray = Nd4j.zeros(numSamples, iter.inputColumns)
      //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
      for (s <- 0 until numSamples) {
        val outputProbDistribution: Array[Double] = new Array[Double](iter.totalOutcomes)
        for (j <- outputProbDistribution.indices) {
          outputProbDistribution(j) = output.getDouble(s, j)
        }
        val sampledCharacterIdx = GravesLSTMCharModellingExample.sampleFromDistribution(outputProbDistribution, rng)
        nextInput.putScalar(Array[Int](s, sampledCharacterIdx), 1.0f) //Prepare next time step input
        sb(s).append(iter.convertIndexToCharacter(sampledCharacterIdx)) //Add sampled character to StringBuilder (human readable output)
      }
      output = net.rnnTimeStep(nextInput)(0) //Do one time step of forward pass
    }
    (0 until numSamples).map({ i => sb(i).toString }).toArray
  }

}
