package org.deeplearning4j.examples.recurrent.word2vecsentiment

import java.io.{File, IOException}
import java.util.NoSuchElementException

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

import scala.collection.JavaConverters._
import scala.collection.mutable

/** This is a DataSetIterator that is specialized for the IMDB review dataset used in the Word2VecSentimentRNN example
 * It takes either the train or test set data from this data set, plus a WordVectors object (typically the Google News
 * 300 pretrained vectors from https://code.google.com/p/word2vec/) and generates training data sets.<br>
 * Inputs/features: variable-length time series, where each word (with unknown words removed) is represented by
 * its Word2Vec vector representation.<br>
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author Alex Black
 */

/**
 * @param dataDirectory the directory of the IMDB review data set
 * @param wordVectors WordVectors object
 * @param batchSize Size of each minibatch for training
 * @param truncateLength If reviews exceed
 * @param train If true: return the training data. If false: return the testing data.
 */
@throws[IOException]
class SentimentExampleIterator(dataDirectory: String, wordVectors: WordVectors, batchSize: Int, truncateLength: Int, train: Boolean) extends DataSetIterator {

    private[this] var cursorPos = 0

    private[this] val vectorSize = wordVectors.lookupTable().layerSize()

    private val positiveFiles = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/pos/") + "/").listFiles
    private val negativeFiles = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/neg/") + "/").listFiles
    private val tokenizerFactory = makeTokenizerFactory

    private[this] def makeTokenizerFactory: TokenizerFactory= {
        val t = new DefaultTokenizerFactory()
        t.setTokenPreProcessor(new CommonPreprocessor())
        t
    }


    @throws[RuntimeException]
    override def next(num: Int): DataSet = {
        if (cursorPos >= positiveFiles.length + negativeFiles.length) throw new NoSuchElementException()
        try {
            nextDataSet(num)
        } catch {
            case e: Exception => throw new RuntimeException(e)
        }
    }

    @throws[IOException]
    private[this] def nextDataSet(num: Int): DataSet = {
        //First: load reviews to String. Alternate positive and negative reviews
        val reviewsBuilder = mutable.ArrayBuilder.make[String]
        val positiveBuilder = mutable.ArrayBuilder.make[Boolean]

        var i = 0
        while (i < num && cursorPos < totalExamples()) {
            if(cursorPos % 2 == 0){
                //Load positive review
                val posReviewNumber = cursorPos / 2
                val review: String = FileUtils.readFileToString(positiveFiles(posReviewNumber))
                reviewsBuilder += review
                positiveBuilder += true
            } else {
                //Load negative review
                val negReviewNumber = cursorPos / 2
                val review: String = FileUtils.readFileToString(negativeFiles(negReviewNumber))
                reviewsBuilder += review
                positiveBuilder += false
            }
            i += 1
            cursorPos += 1
        }
        val reviews = reviewsBuilder.result()
        val positive = positiveBuilder.result()

        //Second: tokenize reviews and filter out unknown words
        var maxLength = 0
        val allTokens = reviews.map { s =>
          val tokens = tokenizerFactory.create(s).getTokens.asScala
          val tokensFiltered = tokens.filter(wordVectors.hasWord)
          maxLength = Math.max(maxLength, tokensFiltered.size)
          tokensFiltered
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        val features = Nd4j.create(reviews.length, vectorSize, maxLength)
        val labels = Nd4j.create(reviews.length, 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        val featuresMask = Nd4j.zeros(reviews.length, maxLength)
        val labelsMask = Nd4j.zeros(reviews.length, maxLength)

        reviews.indices.foreach { i =>
            val tokens = allTokens(i)

            (0 until math.min(tokens.size, maxLength)).foreach { j =>
                val token = tokens(j)
                val vector = wordVectors.getWordVectorMatrix(token)
                features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)), vector)
                featuresMask.putScalar(Array(i, j), 1.0)
            }

            val idx = if (positive(i)) 0 else 1
            val lastIdx = Math.min(tokens.size, maxLength)
            labels.putScalar(Array(i, idx, lastIdx - 1), 1.0)
            labelsMask.putScalar(Array(i, lastIdx - 1), 1.0)
        }
        new DataSet(features,labels,featuresMask,labelsMask)
    }

    override def totalExamples(): Int = positiveFiles.length + negativeFiles.length

    override def inputColumns(): Int = vectorSize

    override def totalOutcomes(): Int = 2

    override def reset() { cursorPos = 0 }

    override def resetSupported(): Boolean = true

    override def batch(): Int = batchSize

    override def cursor(): Int = cursorPos

    override def numExamples(): Int = totalExamples()

    override def setPreProcessor(preProcessor: DataSetPreProcessor) {
        throw new UnsupportedOperationException()
    }

    override def getLabels: java.util.List[String] = {
        java.util.Arrays.asList("positive","negative")
    }

    override def hasNext: Boolean = cursorPos < numExamples()

    override def next(): DataSet = next(batchSize)

    override def remove(): Unit = ()

    override def getPreProcessor(): DataSetPreProcessor = {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** Convenience method for loading review to String */
    @throws[IOException]
    def loadReviewToString(index: Int): String = {
        val f = if (index%2 == 0) {
            positiveFiles(index / 2)
        } else {
            negativeFiles(index / 2)
        }
        FileUtils.readFileToString(f)
    }

    /** Convenience method to get label for review */
    def isPositiveReview(index: Int): Boolean = index % 2 == 0
}
