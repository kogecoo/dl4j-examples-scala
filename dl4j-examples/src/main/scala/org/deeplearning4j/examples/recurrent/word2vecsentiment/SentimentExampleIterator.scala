package org.deeplearning4j.examples.recurrent.word2vecsentiment

import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}

import java.io.{File, IOException}
import java.util
import java.util.NoSuchElementException

import scala.collection.JavaConverters._

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
  * @param dataDirectory  the directory of the IMDB review data set
  * @param wordVectors    WordVectors object
  * @param batchSize      Size of each minibatch for training
  * @param truncateLength If reviews exceed
  * @param train          If true: return the training data. If false: return the testing data.
  */
@throws[IOException]
class SentimentExampleIterator(val dataDirectory: String,
                               val wordVectors: WordVectors,
                               val batchSize: Int,
                               val truncateLength: Int,
                               val train: Boolean) extends DataSetIterator {

  private final val vectorSize = wordVectors.getWordVector(wordVectors.vocab.wordAtIndex(0)).length

  final private val p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/pos/") + "/")
  final private val n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (if (train) "train" else "test") + "/neg/") + "/")
  final private val positiveFiles: Array[File] = p.listFiles
  final private val negativeFiles: Array[File] = n.listFiles
  final private val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory
  tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

  private var _cursor: Int = 0

  def next(num: Int): DataSet = {
    if (_cursor >= positiveFiles.length + negativeFiles.length)
      throw new NoSuchElementException

    try {
      nextDataSet(num)
    } catch { case e: IOException =>
      throw new RuntimeException(e)
    }
  }

  @throws[IOException]
  private def nextDataSet(num: Int): DataSet = {
    //First: load reviews to String. Alternate positive and negative reviews
    val reviews = new util.ArrayList[String](num)
    val positive = new Array[Boolean](num)
    var i = 0
    while (i < num && _cursor < totalExamples) {
      if (_cursor % 2 == 0) {
        //Load positive review
        val posReviewNumber: Int = _cursor / 2
        val review: String = FileUtils.readFileToString(positiveFiles(posReviewNumber))
        reviews.add(review)
        positive(i) = true
      } else {
        //Load negative review
        val negReviewNumber: Int = _cursor / 2
        val review: String = FileUtils.readFileToString(negativeFiles(negReviewNumber))
        reviews.add(review)
        positive(i) = false
      }
      _cursor += 1
      i += 1
    }

    //Second: tokenize reviews and filter out unknown words
    val allTokens: util.List[util.List[String]] = new util.ArrayList[util.List[String]](reviews.size)
    var maxLength: Int = 0
    import scala.collection.JavaConversions._
    for (s <- reviews) {
      val tokens = tokenizerFactory.create(s).getTokens
      val tokensFiltered = new util.ArrayList[String]
      for (t <- tokens.asScala) {
        if (wordVectors.hasWord(t)) tokensFiltered.add(t)
      }
      allTokens.add(tokensFiltered)
      maxLength = Math.max(maxLength, tokensFiltered.size)
    }

    //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    if (maxLength > truncateLength) maxLength = truncateLength

    //Create data for training
    //Here: we have reviews.size() examples of varying lengths
    val features = Nd4j.create(reviews.size, vectorSize, maxLength)
    val labels = Nd4j.create(reviews.size, 2, maxLength)
    //Two labels: positive or negative
    //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
    //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    val featuresMask = Nd4j.zeros(reviews.size, maxLength)
    val labelsMask = Nd4j.zeros(reviews.size, maxLength)

    val temp = new Array[Int](2)
    for (i <- reviews.indices) {
      val tokens: util.List[String] = allTokens.get(i)
      temp(0) = i
      //Get word vectors for each word in review, and put them in the training data
      var j: Int = 0
      for (j <- 0 until math.min(tokens.size, maxLength)) {
        val token = tokens.get(j)
        val vector = wordVectors.getWordVectorMatrix(token)
        features.put(Array[INDArrayIndex](NDArrayIndex.point(i), NDArrayIndex.all, NDArrayIndex.point(j)), vector)

        temp(1) = j
        featuresMask.putScalar(temp, 1.0) //Word is present (not padding) for this example + time step -> 1.0 in features mask
      }

      val idx = if (positive(i)) 0 else 1
      val lastIdx: Int = Math.min(tokens.size, maxLength)
      labels.putScalar(Array[Int](i, idx, lastIdx - 1), 1.0) //Set label: [0,1] for negative, [1,0] for positive
      labelsMask.putScalar(Array[Int](i, lastIdx - 1), 1.0) //Specify that an output exists at the final time step for this example
    }
    new DataSet(features, labels, featuresMask, labelsMask)
  }

  def totalExamples: Int =
    positiveFiles.length + negativeFiles.length

  def inputColumns: Int =
    vectorSize

  def totalOutcomes: Int =
    2

  def reset(): Unit =
    _cursor = 0

  def resetSupported: Boolean =
    true

  def asyncSupported: Boolean =
    true

  def batch: Int =
    batchSize

  def cursor: Int =
    _cursor

  def numExamples: Int =
    totalExamples

  def setPreProcessor(preProcessor: DataSetPreProcessor): Unit =
    throw new UnsupportedOperationException

  def getLabels: util.List[String] =
    util.Arrays.asList("positive", "negative")

  def hasNext: Boolean =
    cursor < numExamples

  def next: DataSet =
    next(batchSize)

  override def remove(): Unit =
    ()

  def getPreProcessor: DataSetPreProcessor =
    throw new UnsupportedOperationException("Not implemented")

  /** Convenience method for loading review to String */
  @throws[IOException]
  def loadReviewToString(index: Int): String = {
    val f = if (index % 2 == 0)
      positiveFiles(index / 2)
    else
      negativeFiles(index / 2)
    FileUtils.readFileToString(f)
  }

  /** Convenience method to get label for review */
  def isPositiveReview(index: Int): Boolean =
    index % 2 == 0

  /**
    * Used post training to load a review from a file to a features INDArray that can be passed to the network output method
    *
    * @param file      File to load the review from
    * @param maxLength Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array
    * @throws IOException If file cannot be read
    */
  @throws[IOException]
  def loadFeaturesFromFile(file: File, maxLength: Int): INDArray = {
    val review = FileUtils.readFileToString(file)
    loadFeaturesFromString(review, maxLength)
  }

  /**
    * Used post training to convert a String to a features INDArray that can be passed to the network output method
    *
    * @param reviewContents Contents of the review to vectorize
    * @param maxLength      Maximum length (if review is longer than this: truncate to maxLength). Use Integer.MAX_VALUE to not nruncate
    * @return Features array for the given input String
    */
  def loadFeaturesFromString(reviewContents: String, maxLength: Int): INDArray = {
    val tokens = tokenizerFactory.create(reviewContents).getTokens
    val tokensFiltered = new util.ArrayList[String]
    for (t <- tokens.asScala) {
      if (wordVectors.hasWord(t)) tokensFiltered.add(t)
    }
    val outputLength = Math.max(maxLength, tokensFiltered.size)

    val features = Nd4j.create(1, vectorSize, outputLength)
    for (j <- 0 until math.min(tokens.size, maxLength)) {
      val token = tokens.get(j)
      val vector = wordVectors.getWordVectorMatrix(token)
      features.put(Array[INDArrayIndex](NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(j)), vector)
    }
    features
  }
}
