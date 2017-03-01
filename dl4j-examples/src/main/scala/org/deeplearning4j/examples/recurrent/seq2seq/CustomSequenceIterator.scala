package org.deeplearning4j.examples.recurrent.seq2seq

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.dataset.api.{MultiDataSet, MultiDataSetPreProcessor}
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable
import scala.util.Random

/**
  * Created by susaneraly on 3/27/16.
  * This is class to generate a multidataset from the AdditionRNN problem
  * Features of the multidataset
  *      - encoder input, eg. "12+13" and
  *      - decoder input, eg. "Go25 " for training and "Go   " for test
  * Labels of the multidataset
  *      - decoder output, "25 End"
  * These strings are encoded as one hot vector sequences.
  *
  * Sequences generated during test are never before seen by the net
  * The random number generator seed is used for repeatability so that each reset of the iterator gives the same data in the same order.
  */
object CustomSequenceIterator {

  private val numDigits = AdditionRNN.NUM_DIGITS
  val SEQ_VECTOR_DIM = AdditionRNN.FEATURE_VEC_SIZE
  val oneHotMap = mutable.Map.empty[String, Integer]
  val oneHotOrder: Array[String] = new Array[String](SEQ_VECTOR_DIM)

  /*
    Takes in an array of strings and return a one hot encoded array of size 1 x 14 x timesteps
    Each element in the array indicates a time step
    Length of one hot vector = 14
  */
  private def mapToOneHot(toEncode: Array[String]): INDArray = {
    val ret: INDArray = Nd4j.zeros(1, SEQ_VECTOR_DIM, toEncode.length)
    for (i <- toEncode.indices) {
      ret.putScalar(0, oneHotMap(toEncode(i)), i, 1)
    }
    ret
  }

  def mapToString(encodeSeq: INDArray, decodeSeq: INDArray): String = {
    mapToString(encodeSeq, decodeSeq, " --> ")
  }

  def mapToString(encodeSeq: INDArray, decodeSeq: INDArray, sep: String): String = {
    val encodeSeqS = oneHotDecode(encodeSeq)
    val decodeSeqS = oneHotDecode(decodeSeq)

    encodeSeqS.indices.map({ i =>
      "\t" + encodeSeqS(i) + sep + decodeSeqS(i) + "\n"
    }).foldLeft("") { case (acc, s) => acc + s }
  }

  /*
   Helper method that takes in a one hot encoded INDArray and returns an interpreted array of strings
   toInterpret size batchSize x one_hot_vector_size(14) x time_steps
   */
  def oneHotDecode(toInterpret: INDArray): Array[String] = {

    val decodedString = new Array[String](toInterpret.size(0))
    val oneHotIndices = Nd4j.argMax(toInterpret, 1) //drops a dimension, so now a two dim array of shape batchSize x time_steps
    for (i <- 0 until oneHotIndices.size(0)) {
      val currentSlice: Array[Int] = oneHotIndices.slice(i).dup.data.asInt //each slice is a batch
      decodedString(i) = mapFromOneHot(currentSlice)
    }
    decodedString
  }

  private def mapFromOneHot(toM: Array[Int]): String = {
    val ret = toM.foldLeft("") { case (acc, m) => acc + oneHotOrder(m) }

    //encoder sequence, needs to be reversed
    if (toM.length > numDigits + 1 + 1) {
      new StringBuilder(ret).reverse.toString
    }
    ret
  }

  /*
   One hot encoding map
   */
  private def oneHotEncoding() {
    for (i <- 0 until 10) {
      oneHotOrder(i) = String.valueOf(i)
      oneHotMap.put(String.valueOf(i), i)
    }
    oneHotOrder(10) = " "
    oneHotMap.put(" ", 10)

    oneHotOrder(11) = "+"
    oneHotMap.put("+", 11)

    oneHotOrder(12) = "Go"
    oneHotMap.put("Go", 12)

    oneHotOrder(13) = "End"
    oneHotMap.put("End", 13)
  }
}

class CustomSequenceIterator(val seed: Int, val batchSize: Int, val totalBatches: Int) extends MultiDataSetIterator {

  private var randnumG: Random = new Random(seed)
  private var seenSequences = mutable.Set.empty[String]
  private var toTestSet: Boolean = false
  private var currentBatch: Int = 0

  CustomSequenceIterator.oneHotEncoding()

  def generateTest(testSize: Int): MultiDataSet = {
    toTestSet = true
    val testData = next(testSize)
    reset()
    testData
  }

  def next(sampleSize: Int): MultiDataSet = {
    var encoderSeq: INDArray = null
    var decoderSeq: INDArray = null
    var outputSeq: INDArray = null

    var currentCount = 0
    var num1 = 0
    var num2 = 0
    val encoderSeqList = mutable.ArrayBuffer.empty[INDArray]
    val decoderSeqList = mutable.ArrayBuffer.empty[INDArray]
    val outputSeqList = mutable.ArrayBuffer.empty[INDArray]

    def genForSum() = {
      num1 = randnumG.nextInt(Math.pow(10, CustomSequenceIterator.numDigits).toInt)
      num2 = randnumG.nextInt(Math.pow(10, CustomSequenceIterator.numDigits).toInt)
      String.valueOf(num1) + "+" + String.valueOf(num2)
    }

    while (currentCount < sampleSize) {
      var forSum = genForSum()
      while (seenSequences.add(forSum)) { forSum = genForSum() }

      val encoderInput = prepToString(num1, num2)
      encoderSeqList += CustomSequenceIterator.mapToOneHot(encoderInput)
      val decoderInput = prepToString(num1 + num2, goFirst = true)
      if (toTestSet) {
        //wipe out everything after "go"; not necessary since we do not use these at test time but here for clarity
        for (i <- 1 until decoderInput.length) {
          decoderInput(i) = " "
        }
      }
      decoderSeqList += CustomSequenceIterator.mapToOneHot(decoderInput)
      val decoderOutput = prepToString(num1 + num2, goFirst = false)
      outputSeqList += CustomSequenceIterator.mapToOneHot(decoderOutput)
      currentCount += 1
    }
    encoderSeq = Nd4j.vstack(encoderSeqList.toArray:_*)
    decoderSeq = Nd4j.vstack(decoderSeqList.toArray:_*)
    outputSeq = Nd4j.vstack(outputSeqList.toArray:_*)
    val inputs = Array[INDArray](encoderSeq, decoderSeq)
    val inputMasks = Array[INDArray](Nd4j.ones(sampleSize, CustomSequenceIterator.numDigits * 2 + 1), Nd4j.ones(sampleSize, CustomSequenceIterator.numDigits + 1 + 1))
    val labels = Array[INDArray](outputSeq)
    val labelMasks = Array[INDArray](Nd4j.ones(sampleSize, CustomSequenceIterator.numDigits + 1 + 1))
    currentBatch += 1
    new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks)
  }

  def reset() {
    currentBatch = 0
    toTestSet = false
    seenSequences = mutable.Set.empty[String]
    randnumG = new Random(seed)
  }

  def resetSupported: Boolean = true

  def asyncSupported: Boolean = false

  def hasNext: Boolean = currentBatch < totalBatches

  def next: MultiDataSet = next(batchSize)

  def remove() {
    throw new UnsupportedOperationException("Not supported")
  }

  def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor) { }

  /*
    Helper method for encoder input
    Given two numbers, num1 and num, returns a string array which represents the input to the encoder RNN
    Note that the string is padded to the correct length and reversed
    Eg. num1 = 7, num 2 = 13 will return {"3","1","+","7"," "}
  */
  def prepToString(num1: Int, num2: Int): Array[String] = {
    val encoded = new Array[String](CustomSequenceIterator.numDigits * 2 + 1)
    var num1S = String.valueOf(num1)
    var num2S = String.valueOf(num2)
    //padding
    while (num1S.length < CustomSequenceIterator.numDigits) {
      num1S = " " + num1S
    }
    while (num2S.length < CustomSequenceIterator.numDigits) {
      num2S = " " + num2S
    }
    val sumString: String = num1S + "+" + num2S
    for (i <- encoded.indices) {
      encoded((encoded.length - 1) - i) = Character.toString(sumString.charAt(i))
    }
    encoded
  }

  /*
      Helper method for decoder input when goFirst
                    for decoder output when !goFirst
      Given a number, return a string array which represents the decoder input (or output) given goFirst (or !goFirst)

      eg. For numDigits = 2 and sum = 31
              if goFirst will return  {"go","3","1", " "}
              if !goFirst will return {"3","1"," ","eos"}
   */
  def prepToString(sum: Int, goFirst: Boolean): Array[String] = {
    var start: Int = 0
    var end: Int = 0
    val decoded: Array[String] = new Array[String](CustomSequenceIterator.numDigits + 1 + 1)
    if (goFirst) {
      decoded(0) = "Go"
      start = 1
      end = decoded.length - 1
    } else {
      start = 0
      end = decoded.length - 2
      decoded(decoded.length - 1) = "End"
    }
    val sumString: String = String.valueOf(sum)
    var maxIndex: Int = start
    //add in digits
    for (i <- sumString.indices) {
      decoded(start + i) = Character.toString(sumString.charAt(i))
      maxIndex += i
    }
    maxIndex += 1
    //needed padding
    while (maxIndex <= end) {
        decoded(maxIndex) = " "
        maxIndex += 1
    }
    decoded
  }
}
