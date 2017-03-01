package org.deeplearning4j.examples.recurrent.basic

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.util
import java.util.Random

/**
  * This example trains a RNN. WHen trained we only have to put the first
  * character of LEARNSTRING to the RNN, and it will recite the following chars
  *
  * @author Peter Grossmann
  */
object BasicRNNExample {
  // define a sentence to learn
  val LEARNSTRING: Array[Char] = "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray

  // a list of all possible characters
  val LEARNSTRING_CHARS_LIST: util.List[Character] = new util.ArrayList[Character]

  // RNN dimensions
  val HIDDEN_LAYER_WIDTH = 50
  val HIDDEN_LAYER_CONT = 2
  val r = new Random(7894)

  def main(args: Array[String]) {

    // create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
    val LEARNSTRING_CHARS: util.LinkedHashSet[Character] = new util.LinkedHashSet[Character]
    for (c <- LEARNSTRING) LEARNSTRING_CHARS.add(c)
    LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS)

    // some common parameters
    val builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder
    builder.iterations(10)
    builder.learningRate(0.001)
    builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    builder.seed(123)
    builder.biasInit(0)
    builder.miniBatch(false)
    builder.updater(Updater.RMSPROP)
    builder.weightInit(WeightInit.XAVIER)

    val listBuilder = builder.list

    // first difference, for rnns we need to use GravesLSTM.Builder
    for (i <- 0 until HIDDEN_LAYER_CONT) {
      val hiddenLayerBuilder: GravesLSTM.Builder = new GravesLSTM.Builder
      hiddenLayerBuilder.nIn(if (i == 0) LEARNSTRING_CHARS.size else HIDDEN_LAYER_WIDTH)
      hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH)

      // adopted activation function from GravesLSTMCharModellingExample
      // seems to work well with RNNs
      hiddenLayerBuilder.activation(Activation.TANH)
      listBuilder.layer(i, hiddenLayerBuilder.build)
    }

    // we need to use RnnOutputLayer for our RNN
    val outputLayerBuilder: RnnOutputLayer.Builder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
    // softmax normalizes the output neurons, the sum of all outputs is 1
    // this is required for our sampleFromDistribution-function
    outputLayerBuilder.activation(Activation.SOFTMAX)
    outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)
    outputLayerBuilder.nOut(LEARNSTRING_CHARS.size)
    listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build)

    // finish builder
    listBuilder.pretrain(false)
    listBuilder.backprop(true)

    // create network
    val conf = listBuilder.build
    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    /*
     * CREATE OUR TRAINING DATA
     */
    // create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
    // SEQUENCE_POSITION
    val input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.length)
    val labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.length)
    // loop through our sample-sentence
    var samplePos = 0
    for (currentChar <- LEARNSTRING) {
      // small hack: when currentChar is the last, take the first char as
      // nextChar - not really required
      val nextChar = LEARNSTRING((samplePos + 1) % (LEARNSTRING.length))
      // input neuron for current-char is 1 at "samplePos"
      input.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos), 1)
      // output neuron for next-char is 1 at "samplePos"
      labels.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos), 1)
      samplePos += 1
    }
    val trainingData: DataSet = new DataSet(input, labels)

    // some epochs
    for (epoch <- 0 until 100) {
      println("Epoch " + epoch)

      // train the data
      net.fit(trainingData)

      // clear current stance from the last example
      net.rnnClearPreviousState()

      // put the first caracter into the rrn as an initialisation
      val testInit: INDArray = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size)
      testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING(0)), 1)

      // run one step -> IMPORTANT: rnnTimeStep() must be called, not
      // output()
      // the output shows what the net thinks what should come next
      var output: INDArray = net.rnnTimeStep(testInit)

      // now the net sould guess LEARNSTRING.length mor characters
      for (j <- LEARNSTRING.indices) {
        // first process the last output of the network to a concrete
        // neuron, the neuron with the highest output cas the highest
        // cance to get chosen
        val outputProbDistribution: Array[Double] = new Array[Double](LEARNSTRING_CHARS.size)
        for (k <- outputProbDistribution.indices) {
          outputProbDistribution(k) = output.getDouble(k)
        }
        val sampledCharacterIdx = findIndexOfHighestValue(outputProbDistribution)

        // print the chosen output
        print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx))

        // use the last output as input
        val nextInput: INDArray = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size)
        nextInput.putScalar(sampledCharacterIdx, 1)
        output = net.rnnTimeStep(nextInput)
      }
      print("\n")
    }
  }

  private def findIndexOfHighestValue(distribution: Array[Double]): Int = {
    var maxValueIndex: Int = 0
    var maxValue: Double = 0
    for (i <- distribution.indices) {
      if (distribution(i) > maxValue) {
        maxValue = distribution(i)
        maxValueIndex = i
      }
    }
    maxValueIndex
  }
}
