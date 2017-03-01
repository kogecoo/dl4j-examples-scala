package org.deeplearning4j.examples.recurrent.character

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util
import java.util.{Collections, Random}

import scala.collection.JavaConverters._

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
  * Given a text file and a few options, generate feature vectors and labels for training,
  * where we want to predict the next character in the sequence.<br>
  * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
  * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
  * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
  *
  * Feature vectors and labels are both one-hot vectors of same length
  *
  * @author Alex Black
  */
object CharacterIterator {
  /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
  def getMinimalCharacterSet: Array[Char] = (
    ('a' to 'z') ++ ('A' to 'Z') ++ ('0' until '9') ++
      Array('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')
    ).toArray

  /** As per getMinimalCharacterSet(), but with a few extra characters */
  def getDefaultCharacterSet: Array[Char] = {
    getMinimalCharacterSet ++
      Array('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')
  }

}


/**
  * @param textFilePath     Path to text file to use for generating samples
  * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
  * @param miniBatchSize    Number of examples per mini-batch
  * @param exampleLength    Number of characters in each input/output vector
  * @param validCharacters  Character array of valid characters. Characters not present in this array will be removed
  * @param rng              Random number generator, for repeatability if required
  * @throws IOException If text file cannot  be loaded
  */
@throws[IOException]
class CharacterIterator(
 val textFilePath: String,
 val textFileEncoding: Charset, //Size of each minibatch (number of examples)
 var miniBatchSize: Int, //Length of each example/minibatch (number of characters)
 var exampleLength: Int, //Valid characters
 var validCharacters: Array[Char], var rng: Random
) extends DataSetIterator {

  if (!new File(textFilePath).exists)
    throw new IOException("Could not access file (does not exist): " + textFilePath)

  if (miniBatchSize <= 0)
    throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)")

  //Maps each character to an index ind the input/output
  //Store valid characters is a map for later use in vectorization
  private val charToIdxMap: Map[Char, Int] =
    validCharacters.indices.map({ i => (validCharacters(i), i)}).toMap

  //Offsets for the start of each example
  private val exampleStartOffsets: util.LinkedList[Integer] = new util.LinkedList[Integer]


  //Load file and convert contents to a char[]
  val lines: util.List[String] = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding)
  val maxSize: Int  = lines.size + lines.asScala.map(_.length).sum //add lines.size() to account for newline characters at end of each line
  val characters = new Array[Char](maxSize)
  var currIdx: Int = 0

  for (s <- lines.asScala) {
    val thisLine = s.toCharArray
    for (aThisLine <- thisLine; if charToIdxMap.isDefinedAt(aThisLine)) {
      characters(currIdx) = aThisLine
      currIdx += 1
    }
    if (charToIdxMap.isDefinedAt('\n')) {
      characters(currIdx) = '\n'
      currIdx += 1
    }
  }

  //All characters of the input file (after filtering to only those that are valid
  private val fileCharacters: Array[Char] = if (currIdx == characters.length) {
    characters
  } else {
    util.Arrays.copyOfRange(characters, 0, currIdx)
  }

  if (exampleLength >= fileCharacters.length)
    throw new IllegalArgumentException("exampleLength=" + exampleLength +
      " cannot exceed number of valid characters in file (" + fileCharacters.length + ")")

  println("Loaded and converted file: " + fileCharacters.length +
    " valid characters of " + maxSize + " total characters (" + (maxSize - fileCharacters.length) + " removed)")

  initializeOffsets()


  def convertIndexToCharacter(idx: Int): Char =
    validCharacters(idx)

  def convertCharacterToIndex(c: Char): Int =
    charToIdxMap(c)

  def getRandomCharacter: Char =
    validCharacters((rng.nextDouble * validCharacters.length).toInt)

  def hasNext: Boolean =
    exampleStartOffsets.size > 0

  def next: DataSet =
    next(miniBatchSize)

  def next(num: Int): DataSet = {
    if (exampleStartOffsets.size == 0) throw new NoSuchElementException
    val currMinibatchSize: Int = Math.min(num, exampleStartOffsets.size)
    //Allocate space:
    //Note the order here:
    // dimension 0 = number of examples in minibatch
    // dimension 1 = size of each vector (i.e., number of characters)
    // dimension 2 = length of each time series/example
    //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
    val input = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length, exampleLength), 'f')
    val labels = Nd4j.create(Array[Int](currMinibatchSize, validCharacters.length, exampleLength), 'f')
    for (i <- 0 until currMinibatchSize) {
      val startIdx = exampleStartOffsets.removeFirst()
      val endIdx = startIdx + exampleLength
      var currCharIdx = charToIdxMap(fileCharacters(startIdx))
      //Current input
      var c: Int = 0
      for (j <- startIdx + 1 until endIdx) {
        val nextCharIdx = charToIdxMap(fileCharacters(j)) //Next character to predict
        input.putScalar(Array[Int](i, currCharIdx, c), 1.0)
        labels.putScalar(Array[Int](i, nextCharIdx, c), 1.0)
        currCharIdx = nextCharIdx
        c += 1
      }
    }
    new DataSet(input, labels)
  }

  def totalExamples: Int =
    (fileCharacters.length - 1) / miniBatchSize - 2

  def inputColumns: Int =
    validCharacters.length

  def totalOutcomes: Int =
    validCharacters.length

  def reset() {
    exampleStartOffsets.clear()
    initializeOffsets()
  }

  private def initializeOffsets(): Unit = {
    //This defines the order in which parts of the file are fetched
    val nMinibatchesPerEpoch: Int = (fileCharacters.length - 1) / exampleLength - 2

    //-2: for end index, and for partial example
    for (i <- 0 until nMinibatchesPerEpoch) {
      exampleStartOffsets.add(i * exampleLength)
    }
    Collections.shuffle(exampleStartOffsets, rng)
  }

  def resetSupported: Boolean =
    true

  def asyncSupported: Boolean =
    true

  def batch: Int =
    miniBatchSize

  def cursor: Int =
    totalExamples - exampleStartOffsets.size

  def numExamples: Int =
    totalExamples

  def setPreProcessor(preProcessor: DataSetPreProcessor) {
    throw new UnsupportedOperationException("Not implemented")
  }

  def getPreProcessor: DataSetPreProcessor = {
    throw new UnsupportedOperationException("Not implemented")
  }

  def getLabels: util.List[String] = {
    throw new UnsupportedOperationException("Not implemented")
  }

  override def remove() {
    throw new UnsupportedOperationException
  }

}
