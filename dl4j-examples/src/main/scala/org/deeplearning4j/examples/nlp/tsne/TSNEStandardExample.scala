package org.deeplearning4j.examples.nlp.tsne

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.plot.BarnesHutTsne
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.slf4j.{Logger, LoggerFactory}

import java.io.File

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Created by agibsonccc on 9/20/14.
  *
  * Dimensionality reduction for high-dimension datasets
  */
object TSNEStandardExample {
  private val log: Logger = LoggerFactory.getLogger(TSNEStandardExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    //STEP 1: Initialization
    val iterations: Int = 100
    //create an n-dimensional array of doubles
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
    val cacheList = mutable.ArrayBuffer.empty[String] //cacheList is a dynamic array of strings used to hold all words

    //STEP 2: Turn text input into a list of words
    log.info("Load & Vectorize data....")
    val wordFile = new ClassPathResource("words.txt").getFile //Open the file
    //Get the data of all unique word vectors
    val vectors = WordVectorSerializer.loadTxt(wordFile)
    val cache = vectors.getSecond
    val weights= vectors.getFirst.getSyn0 //seperate weights of unique words into their own list

    for (i <- 0 until cache.numWords) {
      //seperate strings of words into their own list
      cacheList += cache.wordAtIndex(i)
    }

    //STEP 3: build a dual-tree tsne to use later
    log.info("Build model....")
    val tsne = new BarnesHutTsne.Builder()
      .setMaxIter(iterations).theta(0.5)
      .normalize(false)
      .learningRate(500)
      .useAdaGrad(false)
//      .usePca(false)
      .build

    //STEP 4: establish the tsne values and save them to a file
    log.info("Store TSNE Coordinates for Plotting....")
    val outputFile: String = "target/archive-tmp/tsne-standard-coords.csv"
    new File(outputFile).getParentFile.mkdirs
    tsne.plot(weights, 2, cacheList.toList.asJava, outputFile)
    //This tsne will use the weights of the vectors as its matrix, have two dimensions, use the words strings as
    //labels, and be written to the outputFile created on the previous line
    // Plot Data with gnuplot
    // set datafile separator ","
    // plot 'tsne-standard-coords.csv' using 1:2:3 with labels font "Times,8"
    //!!! Possible error: plot was recently deprecated. Might need to re-do the last line
  }

}
