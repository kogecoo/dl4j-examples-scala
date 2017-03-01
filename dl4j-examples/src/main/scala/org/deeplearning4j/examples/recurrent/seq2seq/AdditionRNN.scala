package org.deeplearning4j.examples.recurrent.seq2seq

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ComputationGraphConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.graph.rnn.{DuplicateToTimeSeriesVertex, LastTimeStepVertex}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Created by susaneraly on 3/27/16.
  */
object AdditionRNN {

      /*
        This example is modeled off the sequence to sequence RNNs described in http://arxiv.org/abs/1410.4615
        Specifically, a sequence to sequence NN is build for the addition operation. Addition is viewed as a translation task.
        For eg. "12+23 " = " 35" with "12+23 " as the input sequence to be translated to the output sequence " 35"
        For a general idea of seq2seq models refer to the image on Pg. 3 in the paper https://arxiv.org/pdf/1406.1078v3

        This example is build using a computation graph with RNN layers.
        Refer here for more details on computation graphs in dl4j
            https://deeplearning4j.org/compgraph
        And here for RNNs
            https://deeplearning4j.org/usingrnns

        There are two RNN layers to this computation graph. The inputs to them are as follows,
        During training:
            - encoder RNN layer:
                   Takes in the addition input string, eg. '12+13'
            - decoder RNN layer:
                   Takes in an input that combines the following two elements
                      1. The output of the very last time step of the encoder
                      2. The shifted 'correct' output of the addition (by appending with a "Go"), 'Go25 '

            which is then trained to fit to the output of the decoder RNN layer, eg '25 '

        During test the inputs are as follows:
            - encoder RNN layer:
                    Takes in the addition input string '12+13'
            - decoder RNN layer:
                   For a time step t takes in an input that combines the following two elements
                      1. The output of the very last time step of the encoder
                      2. The output of the decoder at time step, t-1; For t = 0 input to the decoder is merely "go"

        One hot vectors are used for encoding/decoding (length of one hot vector is 14 for 10 digits and "+"," ",beginning of string and end of string
        10 epochs give ~85% accuracy for 2 digits
        20 epochs give >95% accuracy for 2 digits

        To try out addition for numbers with different number of digits simply change "NUM_DIGITS"
     */

  val NUM_DIGITS: Int = 2
  //Random number generator seed, for reproducability
  val seed: Int = 1234

  //Tweak these to tune the dataset size = batchSize * totalBatches
  var batchSize: Int = 10
  var totalBatches: Int = 500
  var nEpochs: Int = 10
  var nIterations: Int = 1

  //Tweak the number of hidden nodes
  val numHiddenNodes: Int = 128

  //This is the size of the one hot vector
  val FEATURE_VEC_SIZE: Int = 14

  @throws[Exception]
  def main(args: Array[String]) {

    //DataType is set to double for higher precision
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

    //This is a custom iterator that returns MultiDataSets on each call of next - More details in comments in the class
    val iterator = new CustomSequenceIterator(seed, batchSize, totalBatches)

    val configuration: ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .learningRate(0.25)
      .updater(Updater.SGD)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(nIterations).seed(seed)
      .graphBuilder
      //These are the two inputs to the computation graph
      .addInputs("additionIn", "sumOut")
      .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
      //The inputs to the encoder will have size = minibatch x featuresize x timesteps
      //Note that the network only knows of the feature vector size. It does not know how many time steps unless it sees an instance of the data
      .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build, "additionIn")
      //Create a vertex indicating the very last time step of the encoder layer needs to be directed to other places in the comp graph
      .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
      //Create a vertex that allows the duplication of 2d input to a 3d input
      //In this case the last time step of the encoder layer (viz. 2d) is duplicated to the length of the timeseries "sumOut" which is an input to the comp graph
      //Refer to the javadoc for more detail
      .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
      //The inputs to the decoder will have size = size of output of last timestep of encoder (numHiddenNodes) + size of the other input to the comp graph,sumOut (feature vector size)
      .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE + numHiddenNodes).nOut(numHiddenNodes).activation(Activation.SOFTSIGN).build, "sumOut", "duplicateTimeStep").addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation(Activation.SOFTMAX).lossFunction(LossFunctions.LossFunction.MCXENT).build, "decoder")
      .setOutputs("output").pretrain(false).backprop(true)
      .build

    val net: ComputationGraph = new ComputationGraph(configuration)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    //Train model:
    var iEpoch = 0
    val testSize = 20
    val predictor = new Seq2SeqPredicter(net)
    for (iEpoch <- 0 until nEpochs) {
      net.fit(iterator)
      printf("* = * = * = * = * = * = * = * = * = ** EPOCH %d ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n", iEpoch)
      var testData: MultiDataSet = iterator.generateTest(testSize)
      val predictions: INDArray = predictor.output(testData)
      encode_decode_eval(predictions, testData.getFeatures(0), testData.getLabels(0))
      /*
       (Comment/Uncomment) the following block of code to (see/or not see) how the output of the decoder is fed back into the input during test time
       */
      println("Printing stepping through the decoder for a minibatch of size three:")
      testData = iterator.generateTest(3)
      predictor.output(testData, true)
      println("\n* = * = * = * = * = * = * = * = * = ** EPOCH " + (iEpoch + 1) + " COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =")
    }
  }

  private def encode_decode_eval(predictions: INDArray, questions: INDArray, answers: INDArray) {
    val nTests = predictions.size(0)
    var wrong = 0
    var correct = 0
    val questionS = CustomSequenceIterator.oneHotDecode(questions)
    val answersS = CustomSequenceIterator.oneHotDecode(answers)
    val predictionS = CustomSequenceIterator.oneHotDecode(predictions)
    for (iTest <- 0 until nTests) {
      if (answersS(iTest) != predictionS(iTest)) {
        println(questionS(iTest) + " gives " + predictionS(iTest) + " != " + answersS(iTest))
        wrong += 1
      } else {
        println(questionS(iTest) + " gives " + predictionS(iTest) + " == " + answersS(iTest))
        correct += 1
      }
    }
    val randomAcc = Math.pow(10, -1 * (NUM_DIGITS + 1)) * 100
    println("WRONG: " + wrong)
    println("CORRECT: " + correct)
    println("Note randomly guessing digits in succession gives lower than a accuracy of:" + randomAcc + "%")
    println("The digits along with the spaces have to be predicted\n")
  }
}
