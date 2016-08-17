package org.deeplearning4j.examples.recurrent.basic

import java.util.Random

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
 * This example trains a RNN. WHen trained we only have to put the first
 * character of LEARNSTRING to the RNN, and it will recite the following chars
 *
 * @author Peter Grossmann
 */
object BasicRNNExample {

	// define a sentence to learn
	final val LEARNSTRING: Array[Char] = "Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray

	// a list of all possible characters
	final val LEARNSTRING_CHARS_LIST = new java.util.ArrayList[Character]()

	// RNN dimensions
	final val HIDDEN_LAYER_WIDTH = 50
	final val HIDDEN_LAYER_CONT = 2
	final val r = new Random(7894)

	def main(args: Array[String]) {

		// create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
		val LEARNSTRING_CHARS = new java.util.LinkedHashSet[Character]()
		LEARNSTRING.foreach { c =>
			LEARNSTRING_CHARS.add(c)
		}
		LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS)

		// some common parameters
		val builder = new NeuralNetConfiguration.Builder()
		builder.iterations(10)
		builder.learningRate(0.001)
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		builder.seed(123)
		builder.biasInit(0)
		builder.miniBatch(false)
		builder.updater(Updater.RMSPROP)
		builder.weightInit(WeightInit.XAVIER)

		val listBuilder: ListBuilder = builder.list()

		// first difference, for rnns we need to use GravesLSTM.Builder
		Seq.range(0, HIDDEN_LAYER_CONT).foreach { i =>
			val hiddenLayerBuilder = new GravesLSTM.Builder()
			hiddenLayerBuilder.nIn(if (i == 0) LEARNSTRING_CHARS.size() else HIDDEN_LAYER_WIDTH)
			hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH)
			// adopted activation function from GravesLSTMCharModellingExample
			// seems to work well with RNNs
			hiddenLayerBuilder.activation("tanh")
			listBuilder.layer(i, hiddenLayerBuilder.build())
		}

		// we need to use RnnOutputLayer for our RNN
		val outputLayerBuilder = new RnnOutputLayer.Builder(LossFunction.MCXENT)
		// softmax normalizes the output neurons, the sum of all outputs is 1
		// this is required for our sampleFromDistribution-function
		outputLayerBuilder.activation("softmax")
		outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)
		outputLayerBuilder.nOut(LEARNSTRING_CHARS.size())
		listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build())

		// finish builder
		listBuilder.pretrain(false)
		listBuilder.backprop(true)

		// create network
		val conf: MultiLayerConfiguration = listBuilder.build()
		val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
		net.init()
		net.setListeners(new ScoreIterationListener(1))

		/*
		 * CREATE OUR TRAINING DATA
		 */
		// create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
		// SEQUENCE_POSITION
		val input: INDArray = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length)
		val labels: INDArray  = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size(), LEARNSTRING.length)
		// loop through our sample-sentence
		var samplePos = 0
		LEARNSTRING.foreach { currentChar =>
			// small hack: when currentChar is the last, take the first char as
			// nextChar - not really required
			val nextChar: Char = LEARNSTRING((samplePos + 1) % LEARNSTRING.length)
			// input neuron for current-char is 1 at "samplePos"
			input.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos), 1)
			// output neuron for next-char is 1 at "samplePos"
			labels.putScalar(Array[Int](0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos), 1)
			samplePos += 1
			()
		}
		val trainingData = new DataSet(input, labels)

		// some epochs
		Seq.range(0, 100).foreach { epoch =>

			println("Epoch " + epoch)

			// train the data
			net.fit(trainingData)

			// clear current stance from the last example
			net.rnnClearPreviousState()

			// put the first caracter into the rrn as an initialisation
			val testInit: INDArray = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size())
			testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING(0)), 1)

			// run one step -> IMPORTANT: rnnTimeStep() must be called, not
			// output()
			// the output shows what the net thinks what should come next
			var output: INDArray = net.rnnTimeStep(testInit)

			// now the net sould guess LEARNSTRING.length mor characters
			Seq.range(0, LEARNSTRING.length).foreach { j =>

				// first process the last output of the network to a concrete
				// neuron, the neuron with the highest output cas the highest
				// cance to get chosen
				val outputProbDistribution = Array.range(0, LEARNSTRING_CHARS.size()).map(output.getDouble)
				val sampledCharacterIdx: Int = findIndexOfHighestValue(outputProbDistribution)

				// print the chosen output
				print(LEARNSTRING_CHARS_LIST.get(sampledCharacterIdx))

				// use the last output as input
				val nextInput = Nd4j.zeros(LEARNSTRING_CHARS_LIST.size())
				nextInput.putScalar(sampledCharacterIdx, 1)
				output = net.rnnTimeStep(nextInput)

			}
			print("\n")

		}

	}

	private def findIndexOfHighestValue(distribution: Array[Double]): Int = {
		var maxValueIndex: Int = 0
		var maxValue: Double = 0
		Seq.range(0, distribution.length).foreach { i =>
			if(distribution(i) > maxValue) {
				maxValue = distribution(i)
				maxValueIndex = i
			}
		}
		maxValueIndex
	}

}
