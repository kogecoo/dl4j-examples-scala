package org.deeplearning4j.examples.recurrent.character

import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.{Collections, LinkedList, NoSuchElementException, Random}

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j

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


/**
* @param textFilePath Path to text file to use for generating samples
* @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
* @param miniBatchSize Number of examples per mini-batch
* @param exampleLength Number of characters in each input/output vector
* @param validCharacters Character array of valid characters. Characters not present in this array will be removed
* @param rng Random number generator, for repeatability if required
* @throws IOException If text file cannot  be loaded
*/
class CharacterIterator(textFilePath: String, textFileEncoding: Charset, miniBatchSize: Int, exampleLength: Int, validCharacters: Array[Char], rng: Random) extends DataSetIterator {

  val charToIdxMap: java.util.Map[Character, Integer] = new java.util.HashMap()
  var fileCharacters: Array[Char] = Array[Char]()
  var examplesSoFar = 0

  initValidation(textFilePath, miniBatchSize)
  init(textFilePath, textFileEncoding, miniBatchSize, validCharacters)

  private[this] val exampleStartOffsets: LinkedList[Integer] = new LinkedList();

  private[this] def initValidation(textFilePath: String, miniBatchSize: Int) {
    if ( !new File(textFilePath).exists()) {
      val msg = s"Could not access file (does not exist): $textFilePath"
      throw new IOException(msg)
    }

    if (miniBatchSize <= 0) {
      val msg = "Invalid miniBatchSize (must be >0)"
      throw new IllegalArgumentException(msg)
    }
  }

  private[this] def init(textFilePath: String, textFileEncoding: Charset, miniBatchSize: Int, validCharacters: Array[Char]) {

    //Store valid characters is a map for later use in vectorization
    validCharacters.zipWithIndex.foreach { case (ch, i) => charToIdxMap.put(ch, i) }

    //Load file and convert contents to a char[]
    val newLineValid: Boolean = charToIdxMap.containsKey('\n')
    val lines = Files.readAllLines(new File(textFilePath).toPath, textFileEncoding).asScala
    val maxSize: Int = lines.map(_.length).fold(lines.size)(_ + _ ) //add lines.size() to account for newline characters at end of each line
    fileCharacters = lines.flatMap({ s =>
      val filtered = s.filter(charToIdxMap.containsKey(_)).toString
      if (newLineValid) filtered + "\n" else filtered
    }).toArray

    if (exampleLength >= fileCharacters.length) {
      val msg = s"exampleLength=$exampleLength cannot exceed number of valid characters in file (${fileCharacters.length})"
      throw new IllegalArgumentException(msg)
    }

    val nRemoved = maxSize - fileCharacters.length
    val msg = s"Loaded and converted file: ${fileCharacters.length} valid characters of ${maxSize} total characters (${nRemoved}) removed"
    println(msg)

    //This defines the order in which parts of the file are fetched
    val nMinibatchesPerEpoch = (fileCharacters.length-1) / exampleLength - 2   //-2: for end index, and for partial example
    (0 until nMinibatchesPerEpoch).foreach { i =>
        exampleStartOffsets.add(i * exampleLength)
    }
    Collections.shuffle(exampleStartOffsets,rng)
  }

  def convertIndexToCharacter(idx: Int): Char = validCharacters(idx)

  def convertCharacterToIndex(c: Char): Int = charToIdxMap.get(c)

  def getRandomCharacter: Char = validCharacters((rng.nextDouble()*validCharacters.length).toInt)

  def hasNext: Boolean = exampleStartOffsets.size() > 0

  def next(): DataSet = next(miniBatchSize)

  def next(num: Int): DataSet = {
    if (exampleStartOffsets.size() == 0) throw new NoSuchElementException()

    val currMinibatchSize = Math.min(num, exampleStartOffsets.size())

    //Note the order here:
    // dimension 0 = number of examples in minibatch
    // dimension 1 = size of each vector (i.e., number of characters)
    // dimension 2 = length of each time series/example
    val input = Nd4j.zeros(currMinibatchSize,validCharacters.length,exampleLength)
    val labels = Nd4j.zeros(currMinibatchSize,validCharacters.length,exampleLength)

    (0 until currMinibatchSize).foreach { i =>
        val startIdx = exampleStartOffsets.removeFirst()
        val endIdx = startIdx + exampleLength
        var currCharIdx = charToIdxMap.get(fileCharacters(startIdx))	//Current input
        var c=0
        (startIdx+1 until endIdx).foreach { j =>
            val nextCharIdx = charToIdxMap.get(fileCharacters(j))		//Next character to predict
            input.putScalar(Array[Int](i,currCharIdx,c), 1.0)
            labels.putScalar(Array[Int](i,nextCharIdx,c), 1.0)
            currCharIdx = nextCharIdx
            c += 1
        }
    }
    new DataSet(input,labels)
  }

  def totalExamples(): Int = (fileCharacters.length-1) / miniBatchSize - 2

  def inputColumns(): Int = validCharacters.length

  def totalOutcomes(): Int = validCharacters.length

  def reset(): Unit = {
      exampleStartOffsets.clear()
      val nMinibatchesPerEpoch = totalExamples()
      (0 until nMinibatchesPerEpoch).foreach { i =>
          exampleStartOffsets.add(i * miniBatchSize)
      }
      Collections.shuffle(exampleStartOffsets,rng)
  }

  def batch(): Int = miniBatchSize

  def cursor(): Int = totalExamples() - exampleStartOffsets.size()

  def numExamples(): Int = totalExamples()

  def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException("Not implemented")

  override def getLabels: java.util.List[String] = throw new UnsupportedOperationException("Not implemented")

  override def remove(): Unit = throw new UnsupportedOperationException()

}

object CharacterIterator {

  /** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
  def getMinimalCharacterSet: Array[Char] = {
    (('a' to 'z').toSeq ++
    ('A' to 'Z').toSeq ++
    ('0' to '9').toSeq ++
    Seq[Char]('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')).toArray
  }

  /** As per getMinimalCharacterSet(), but with a few extra characters */
  def getDefaultCharacterSet: Array[Char] = {
    (getMinimalCharacterSet ++
    Seq[Char]('@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>')).toArray
  }

}
