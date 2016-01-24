package org.deeplearning4j.examples.rnn

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.{HashMap, Map, NoSuchElementException, Random}

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._

/** A very simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file to start the sequence and
 * (optionally) scanning backwards to a new line (to ensure we don't start half way through a word
 * for example).<br>
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */


/**
* @param textFilePath Path to text file to use for generating samples
* @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
* @param miniBatchSize Number of examples per mini-batch
* @param exampleLength Number of characters in each input/output vector
* @param numExamplesToFetch Total number of examples to fetch (must be multiple of miniBatchSize). Used in hasNext() etc methods
* @param validCharacters Character array of valid characters. Characters not present in this array will be removed
* @param rng Random number generator, for repeatability if required
* @param alwaysStartAtNewLine if true, scan backwards until we find a new line character (up to MAX_SCAN_LENGTH in case
*  of no new line characters, to avoid scanning entire file)
* @throws IOException If text file cannot  be loaded
*/
class CharacterIterator(textFilePath: String, textFileEncoding: Charset, miniBatchSize: Int, exampleLength: Int, numExamplesToFetch: Int, validCharacters: Array[Char], rng: Random, alwaysStartAtNewLine: Boolean) extends DataSetIterator {

  val serialVersionUID: Long = -7287833919126626356L
  val MAX_SCAN_LENGTH: Int = 200
  val charToIdxMap: Map[Character, Integer] = new HashMap()
  var fileCharacters: Array[Char] = Array[Char]()
  var examplesSoFar = 0
  val numCharacters: Int = validCharacters.length

  initValidation(textFilePath, numExamplesToFetch, miniBatchSize)
  init(textFilePath, textFileEncoding, miniBatchSize, numExamplesToFetch, validCharacters)

  def this(path: String, miniBatchSize: Int, exampleSize: Int, numExamplesToFetch: Int) {
    this(path, Charset.defaultCharset(), miniBatchSize, exampleSize, numExamplesToFetch, CharacterIterator.getDefaultCharacterSet(), new Random(), true)
  }

  private[this] def initValidation(textFilePath: String, numExamplesToFetch: Int, miniBatchSize: Int) {
    if ( !new File(textFilePath).exists()) {
      val msg = s"Could not access file (does not exist): ${textFilePath}"
      throw new IOException(msg)
    }

    if (numExamplesToFetch % miniBatchSize != 0) {
      val msg = "numExamplesToFetch must be a multiple of miniBatchSize"
      throw new IllegalArgumentException(msg)
    }

    if (miniBatchSize <= 0) {
      val msg = "Invalid miniBatchSize (must be >0)"
      throw new IllegalArgumentException(msg)
    }
  }

  private[this] def init(textFilePath: String, textFileEncoding: Charset, miniBatchSize: Int, numExamplesToFetch: Int, validCharacters: Array[Char]) {

    //Store valid characters is a map for later use in vectorization
    validCharacters.zipWithIndex.foreach { case (ch, i) => charToIdxMap.put(ch, i) }

    //Load file and convert contents to a char[] 
    val newLineValid: Boolean = charToIdxMap.containsKey('\n')
    val lines = Files.readAllLines(new File(textFilePath).toPath(), textFileEncoding).asScala
    val maxSize: Int = lines.map(_.length).fold(lines.size)(_ + _ ) //add lines.size() to account for newline characters at end of each line 
    fileCharacters = lines.flatMap({ s =>
      val filtered = s.filter(charToIdxMap.containsKey(_)).toString
      if (newLineValid) filtered + "\n" else filtered
    }).toArray

    if (exampleLength >= fileCharacters.length) {
      val msg = s"exampleLength=${exampleLength} cannot exceed number of valid characters in file (${fileCharacters.length})"
      throw new IllegalArgumentException(msg)
    }

    val nRemoved = maxSize - fileCharacters.length
    val msg = s"Loaded and converted file: ${fileCharacters.length} valid characters of ${maxSize} total characters (${nRemoved}) removed"
    println(msg)
  }

  def convertIndexToCharacter(idx: Int): Char = validCharacters(idx)

  def convertCharacterToIndex(c: Char): Int = charToIdxMap.get(c)

  def getRandomCharacter(): Char = validCharacters((rng.nextDouble()*validCharacters.length).toInt)

  def hasNext(): Boolean = examplesSoFar + miniBatchSize <= numExamplesToFetch

  def next(): DataSet = next(miniBatchSize)

  def next(num: Int): DataSet = {
    if (examplesSoFar+num > numExamplesToFetch) throw new NoSuchElementException()

    //Allocate space:
    val input: INDArray = Nd4j.zeros(Array[Int](num, numCharacters, exampleLength): _*)
    val labels: INDArray = Nd4j.zeros(Array[Int](num, numCharacters, exampleLength): _*)

    val maxStartIdx: Int = fileCharacters.length - exampleLength

    //Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
    // of the file in the same minibatch
    (0 until num).foreach { i =>
      var startIdx = (rng.nextDouble()*maxStartIdx).toInt
      var endIdx = startIdx + exampleLength
      var scanLength = 0
      if (alwaysStartAtNewLine) {
        while (startIdx >= 1 && fileCharacters(startIdx-1) != '\n' && scanLength < MAX_SCAN_LENGTH) {
          startIdx -= 1
          endIdx -= 1
          scanLength += 1
        }
      }

      var currCharIdx = charToIdxMap.get(fileCharacters(startIdx));  //Current input
      var c = 0
      (startIdx+1 to endIdx).foreach { j =>
        val nextCharIdx: Int = charToIdxMap.get(fileCharacters(j));    //Next character to predict
        input.putScalar(Array[Int](i, currCharIdx, c), 1.0)
        labels.putScalar(Array[Int](i, nextCharIdx, c), 1.0)
        currCharIdx = nextCharIdx
        c += 1
      }
    }

    examplesSoFar += num
    return new DataSet(input,labels)
  }

  def totalExamples(): Int = numExamplesToFetch

  def inputColumns(): Int = numCharacters

  def totalOutcomes(): Int = numCharacters

  def reset(): Unit = examplesSoFar = 0

  def batch(): Int = miniBatchSize

  def cursor(): Int = examplesSoFar

  def numExamples(): Int = numExamplesToFetch

  def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException("Not implemented")

  override def remove(): Unit = throw new UnsupportedOperationException()

}

object CharacterIterator {

  /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
  def getMinimalCharacterSet(): Array[Char] = {
    (('a' to 'z').toSeq ++
    ('A' to 'Z').toSeq ++
    ('0' to '9').toSeq ++
    Seq[Char]('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')).toArray
  }

  /** As per getMinimalCharacterSet(), but with a few extra characters */
  def getDefaultCharacterSet(): Array[Char] = {
    (getMinimalCharacterSet() ++
    Seq[Char]('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')).toArray
  }

}
