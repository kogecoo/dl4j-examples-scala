package org.deeplearning4j.examples.recurrent.seq2seq

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

import java.util.ArrayList


/**
 * Created by susaneraly on 3/27/16.
 */
class AdditionRNN {

    /*
        This example is modeled off the sequence to sequence RNNs described in http://arxiv.org/abs/1410.4615
        Specifically, a sequence to sequence NN is build for the addition operation
        Two numbers and the addition operator are encoded as a sequence and passed through an "encoder" RNN
        The output from the last time step of the encoder RNN is reinterpreted as a time series and passed through the "decoder" RNN
        The result is the output of the decoder RNN which in training is the sum, encoded as a sequence.
        One hot vectors are used for encoding/decoding
        20 epochs give >85% accuracy for 2 digits
        To try out addition for numbers with different number of digits simply change "NUM_DIGITS"
     */

    //Random number generator seed, for reproducability
    final val seed = 1234

    final val NUM_DIGITS =2
    final val FEATURE_VEC_SIZE = 12

    //Tweak these to tune - dataset size = batchSize * totalBatches
    final val batchSize = 10
    final val totalBatches = 500
    final val nEpochs: Int = 50
    final val nIterations = 1
    final val numHiddenNodes = 128

    //Currently the sequences are implemented as length = max length
    //This is a placeholder for an enhancement
    final val timeSteps = NUM_DIGITS * 2 + 1

    @throws[Exception]
    def main(args: Array[String]) {

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
        //Training data iterator
        val iterator: CustomSequenceIterator = new CustomSequenceIterator(seed, batchSize, totalBatches, NUM_DIGITS,timeSteps)

        val configuration: ComputationGraphConfiguration = new NeuralNetConfiguration.Builder()
                //.regularization(true).l2(0.000005)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.5)
                .updater(Updater.RMSPROP)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nIterations)
                .seed(seed)
                .graphBuilder()
                .addInputs("additionIn", "sumOut")
                .setInputTypes(InputType.recurrent(FEATURE_VEC_SIZE), InputType.recurrent(FEATURE_VEC_SIZE))
                .addLayer("encoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE).nOut(numHiddenNodes).activation("softsign").build(),"additionIn")
                .addVertex("lastTimeStep", new LastTimeStepVertex("additionIn"), "encoder")
                .addVertex("duplicateTimeStep", new DuplicateToTimeSeriesVertex("sumOut"), "lastTimeStep")
                .addLayer("decoder", new GravesLSTM.Builder().nIn(FEATURE_VEC_SIZE+numHiddenNodes).nOut(numHiddenNodes).activation("softsign").build(), "sumOut","duplicateTimeStep")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(numHiddenNodes).nOut(FEATURE_VEC_SIZE).activation("softmax").lossFunction(LossFunctions.LossFunction.MCXENT).build(), "decoder")
                .setOutputs("output")
                .pretrain(false).backprop(true)
                .build()

        val net: ComputationGraph = new ComputationGraph(configuration)
        net.init()
        //net.setListeners(new ScoreIterationListener(200),new HistogramIterationListener(200))
        net.setListeners(new ScoreIterationListener(1))
        //net.setListeners(new HistogramIterationListener(200))
        //Train model:
        val testSize = 200
        (0 until nEpochs).foreach { i =>
            printf("* = * = * = * = * = * = * = * = * = ** EPOCH %d ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n",i)
            net.fit(iterator)
            val testData: MultiDataSet = iterator.generateTest(testSize)
            val testNums: ArrayList[Array[Int]] = iterator.testFeatures()
            val testnum1: Array[Int] = testNums.get(0)
            val testnum2: Array[Int] = testNums.get(1)
            val testSums: Array[Int] = iterator.testLabels()
            val prediction_array: Array[INDArray]  = net.output(Array[INDArray](testData.getFeatures(0), testData.getFeatures(1)):_*)
            val predictions: INDArray = prediction_array.head
            val answers: INDArray = Nd4j.argMax(predictions,1)

            encode_decode(testnum1,testnum2,testSums,answers)

            iterator.reset()
        }
        printf("\n* = * = * = * = * = * = * = * = * = ** EPOCH COMPLETE ** = * = * = * = * = * = * = * = * = * = * = * = * = * =\n")

    }

    //This is a helper function to make the predictions from the net more readable
    private def encode_decode(num1: Array[Int], num2: Array[Int], sum: Array[Int], answers: INDArray) {

        val nTests = answers.size(0)
        var wrong = 0
        var correct = 0
        (0 until nTests).foreach { iTest =>
            var aDigit = NUM_DIGITS
            var thisAnswer = 0
            var strAnswer = ""

            while (aDigit >= 0) {
                //System.out.println("while"+aDigit+strAnwer)
                val thisDigit = answers.getDouble(iTest,aDigit).asInstanceOf[Int]
                //System.out.println(thisDigit)
                if (thisDigit <= 9) {
                    strAnswer += String.valueOf(thisDigit)
                    thisAnswer += thisDigit * Math.pow(10,aDigit).asInstanceOf[Int]
                } else {
                    //System.out.println(thisDigit+" is string " + String.valueOf(thisDigit))
                    strAnswer += " "
                    //break
                }
                aDigit -= 1
            }
            val strAnswerR = new StringBuilder(strAnswer).reverse.toString.replaceAll("\\s+","")

            if (strAnswerR.equals(String.valueOf(sum(iTest)))) {
                println(num1(iTest)+"+"+num2(iTest)+"=="+strAnswerR)
                correct += 1
            } else {
                println(num1(iTest)+"+"+num2(iTest)+"!="+strAnswerR+", should=="+sum(iTest))
                wrong += 1
            }
        }
        val randomAcc = Math.pow(10,-1*(NUM_DIGITS+1)) * 100
        println("*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*")
        println("WRONG: "+wrong)
        println("CORRECT: "+correct)
        println("Note randomly guessing digits in succession gives lower than a accuracy of:"+randomAcc+"%")
        println("The digits along with the spaces have to be predicted\n")
    }

}

