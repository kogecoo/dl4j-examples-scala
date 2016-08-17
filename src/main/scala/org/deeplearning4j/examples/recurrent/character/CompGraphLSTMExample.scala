package org.deeplearning4j.examples.recurrent.character

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.util.Random

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
class CompGraphLSTMExample {

    @throws[Exception]
    def main(args: Array[String]) {
        val lstmLayerSize = 200					//Number of units in each GravesLSTM layer
        val miniBatchSize = 32						//Size of mini batch to use when  training
        val exampleLength = 1000					//Length of each training example sequence to use. This could certainly be increased
        val tbpttLength = 50                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        val numEpochs = 1							//Total number of training epochs
        val generateSamplesEveryNMinibatches = 10  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        val nSamplesToGenerate = 4					//Number of samples to generate after each training epoch
        val nCharactersToSample = 300				//Length of each sample to generate
        val generationInitialization = null;		//Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        val rng = new Random(12345)

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        val iter: CharacterIterator = GravesLSTMCharModellingExample.getShakespeareIterator(miniBatchSize, exampleLength)
        val nOut = iter.totalOutcomes()

        //Set up network configuration:
        val conf: ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .rmsDecay(0.95)
            .seed(12345)
            .regularization(true)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("input") //Give the input a name. For a ComputationGraph with multiple inputs, this also defines the input array orders
            //First layer: name "first", with inputs from the input called "input"
            .addLayer("first", new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .updater(Updater.RMSPROP).activation("tanh").build(),"input")
            //Second layer, name "second", with inputs from the layer called "first"
            .addLayer("second", new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .updater(Updater.RMSPROP)
                .activation("tanh").build(),"first")
            //Output layer, name "outputlayer" with inputs from the two layers called "first" and "second"
            .addLayer("outputLayer", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation("softmax").updater(Updater.RMSPROP)
                .nIn(2*lstmLayerSize).nOut(nOut).build(),"first","second")
            .setOutputs("outputLayer")  //List the output. For a ComputationGraph with multiple outputs, this also defines the input array orders
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build()

        val net: ComputationGraph = new ComputationGraph(conf)
        net.init()
        net.setListeners(new ScoreIterationListener(1))

        //Print the  number of parameters in the network (and for each layer)
        var totalNumParams = 0
        (0 until net.getNumLayers).foreach { i =>
            val nParams = net.getLayer(i).numParams()
            println("Number of parameters in layer " + i + ": " + nParams)
            totalNumParams += nParams
        }
        println("Total number of network parameters: " + totalNumParams)

        //Do training, and then generate and print samples from network
        var miniBatchNumber = 0
        (0 until numEpochs).foreach { i =>
            while(iter.hasNext()){
                val ds: DataSet = iter.next()
                net.fit(ds)
                miniBatchNumber += 1
                if(miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    println("--------------------")
                    println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" )

                    // generationInitialization is always null, (perhaps this is a programming mistake).
                    val givenInitialization: String = "" // if (generationInitialization == null) "" else generationInitialization

                    println("Sampling characters from network given initialization \"" + givenInitialization + "\"")
                    val samples: Array[String] = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate)
                    samples.indices.foreach { j =>
                        println("----- Sample " + j + " -----")
                        println(samples(j))
                        println()
                    }
                }
            }

            iter.reset();	//Reset iterator for another epoch
        }

        println("\n\nExample complete")
    }

    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    private[this] def sampleCharactersFromNetwork(initialization: String, net: ComputationGraph,
      iter: CharacterIterator, rng: Random, charactersToSample: Int, numSamples: Int): Array[String] = {
        //Set up initialization. If no initialization: use a random character
        val init = if( initialization == null ){
            String.valueOf(iter.getRandomCharacter)
        } else initialization

        //Create input for initialization
        val initializationInput: INDArray = Nd4j.zeros(numSamples, iter.inputColumns(), init.length())
        val initChars: Array[Char] = init.toCharArray
        initChars.indices.foreach { i =>
            val idx: Int = iter.convertCharacterToIndex(initChars(i))
            (0 until numSamples).foreach { j =>
                initializationInput.putScalar(Array(j,idx,i), 1.0f)
            }
        }

        val sb = Array.fill(numSamples)(new StringBuilder)
        (0 until numSamples).foreach { i =>  sb(i) = new StringBuilder(init) }

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState()
        var output: INDArray = net.rnnTimeStep(initializationInput).head
        output = output.tensorAlongDimension(output.size(2)-1,1,0)	//Gets the last time step output

        (0 until charactersToSample).foreach { i =>
            //Set up next input (single time step) by sampling from previous output
            val nextInput: INDArray = Nd4j.zeros(numSamples,iter.inputColumns())
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            (0 until numSamples).foreach { s =>
                val outputProbDistribution = (0 until iter.totalOutcomes()).map({ j => output.getDouble(s, j) }).toArray
                val sampledCharacterIdx = GravesLSTMCharModellingExample.sampleFromDistribution(outputProbDistribution, rng)

                nextInput.putScalar(Array(s,sampledCharacterIdx), 1.0f)		//Prepare next time step input
                sb(s).append(iter.convertIndexToCharacter(sampledCharacterIdx))	//Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput).head	//Do one time step of forward pass
        }

        (0 until numSamples).map({ i => sb(i).toString }).toArray
    }
}
