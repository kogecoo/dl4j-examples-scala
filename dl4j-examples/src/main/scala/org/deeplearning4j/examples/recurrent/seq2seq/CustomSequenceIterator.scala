package org.deeplearning4j.examples.recurrent.seq2seq

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.MultiDataSet
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j

import java.util.ArrayList
import java.util.Random


/**
 * Created by susaneraly on 3/27/16.
 * This is class to generate pairs of random numbers given a maximum number of digits
 * This class can also be used as a reference for dataset iterators and writing one's own custom dataset iterator
 */

class CustomSequenceIterator(seed: Int, batchSize: Int, totalBatches: Int, numdigits: Int, timestep: Int) extends MultiDataSetIterator {

    private[this] val encoderSeqLength: Int = numdigits * 2 + 1
    private[this] val decoderSeqLength: Int = numdigits + 1 + 1
    private[this] val outputSeqLength: Int = numdigits + 1 + 1

    private[this] var num1Arr: Array[Int] = null
    private[this] var num2Arr: Array[Int] = null
    private[this] var sumArr: Array[Int] = null
    private[this] var randnumG: Random = new Random(seed)
    private[this] var currentBatch: Int = 0
    private[this] var toTestSet: Boolean = false

    private final val SEQ_VECTOR_DIM = 12

    def generateTest(testSize: Int): MultiDataSet = {
        toTestSet = true
        next(testSize)
    }

    def testFeatures (): ArrayList[Array[Int]] = {
        val testNums = new ArrayList[Array[Int]]()
        testNums.add(num1Arr)
        testNums.add(num2Arr)
        testNums
    }

    def testLabels(): Array[Int] = sumArr

    override def next(sampleSize: Int): MultiDataSet = {
        /* PLEASE NOTE:
            I don't check for repeats from pair to pair with the generator
            Enhancement, to be fixed later
         */
        //Initialize everything with zeros - will eventually fill with one hot vectors
        val encoderSeq: INDArray = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, encoderSeqLength )
        val decoderSeq: INDArray = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, decoderSeqLength )
        val outputSeq: INDArray = Nd4j.zeros(sampleSize, SEQ_VECTOR_DIM, outputSeqLength )

        //Since these are fixed length sequences of timestep
        //Masks are not required
        val encoderMask: INDArray = Nd4j.ones(sampleSize, encoderSeqLength)
        val decoderMask: INDArray = Nd4j.ones(sampleSize, decoderSeqLength)
        val outputMask: INDArray = Nd4j.ones(sampleSize, outputSeqLength)

        if (toTestSet) {
            num1Arr = Array.fill(sampleSize)(0)
            num2Arr = Array.fill(sampleSize)(0)
            sumArr = Array.fill(sampleSize)(0)
        }

        /* ========================================================================== */
        (0 until sampleSize).foreach { iSample =>
            //Generate two random numbers with numdigits
            val num1 = randnumG.nextInt(Math.pow(10,numdigits).asInstanceOf[Int])
            val num2 = randnumG.nextInt(Math.pow(10,numdigits).asInstanceOf[Int])
            val sum = num1 + num2
            if (toTestSet) {
                num1Arr(iSample) = num1
                num2Arr(iSample) = num2
                sumArr(iSample) = sum
            }
            /*
            Encoder sequence:
            Eg. with numdigits=4, num1=123, num2=90
                123 + 90 is encoded as "   09+321"
                Converted to a string to a fixed size given by 2*numdigits + 1 (for operator)
                then reversed and then masked
                Reversing input gives significant gain
                Each character is transformed to a 12 dimensional one hot vector
                    (index 0-9 for corresponding digits, 10 for "+", 11 for " ")
            */
            var spaceFill = (encoderSeqLength) - (num1 + "+" + num2).length()
            var iPos = 0
            //Fill in spaces, as necessary
            while (spaceFill > 0) {
                //spaces encoded at index 12
                encoderSeq.putScalar(Array(iSample,11,iPos),1)
                iPos += 1
                spaceFill -= 1
            }

            //Fill in the digits in num2 backwards
            val num2Str: String = String.valueOf(num2)
            Range(num2Str.length()-1, -1, -1).foreach { i =>
                val onehot = Character.getNumericValue(num2Str.charAt(i))
                encoderSeq.putScalar(Array(iSample,onehot,iPos),1)
                iPos += 1
            }
            //Fill in operator in this case "+", encoded at index 11
            encoderSeq.putScalar(Array(iSample,10,iPos),1)
            iPos += 1
            //Fill in the digits in num1 backwards
            val num1Str: String = String.valueOf(num1)
            Range(num1Str.length()-1, -1, -1).foreach { i =>
                val onehot = Character.getNumericValue(num1Str.charAt(i))
                encoderSeq.putScalar(Array(iSample,onehot,iPos),1)
                iPos += 1
            }
            //Mask input for rest of the time series
            //while (iPos < timestep) {
            //    encoderMask.putScalar(new []{iSample,iPos},1)
            //    iPos++
            // }
            /*
            Decoder and Output sequences:
            */
            //Fill in the digits from the sum
            iPos = 0
            val sumCharArr: Array[Char] = String.valueOf(num1+num2).toCharArray
            sumCharArr.foreach { c =>
                val digit = Character.getNumericValue(c)
                outputSeq.putScalar(Array(iSample,digit,iPos),1)
                //decoder input filled with spaces
                decoderSeq.putScalar(Array(iSample,11,iPos),1)
                iPos += 1
            }
            //Fill in spaces, as necessary
            //Leaves last index for "."
            while (iPos < numdigits + 1) {
                //spaces encoded at index 12
                outputSeq.putScalar(Array(iSample,11,iPos), 1)
                //decoder input filled with spaces
                decoderSeq.putScalar(Array(iSample,11,iPos),1)
                iPos += 1
            }
            //Predict final " "
            outputSeq.putScalar(Array(iSample,10,iPos), 1)
            decoderSeq.putScalar(Array(iSample,11,iPos), 1)
        }
        //Predict "."
        /* ========================================================================== */
        val inputs = Array[INDArray](encoderSeq, decoderSeq)
        val inputMasks = Array[INDArray](encoderMask, decoderMask)
        val labels = Array[INDArray](outputSeq)
        val labelMasks = Array[INDArray](outputMask)
        currentBatch += 1
        new org.nd4j.linalg.dataset.MultiDataSet(inputs, labels, inputMasks, labelMasks)
    }

    override def reset() {
        currentBatch = 0
        toTestSet = false
        randnumG = new Random(seed)
    }

    def resetSupported(): Boolean = true

    override def hasNext: Boolean = {
        //This generates numbers on the fly
        currentBatch < totalBatches
    }

    override def next(): MultiDataSet = next(batchSize)

    override def remove() {
        throw new UnsupportedOperationException("Not supported")
    }

    def setPreProcessor(multiDataSetPreProcessor: MultiDataSetPreProcessor) {

    }
}

