package org.deeplearning4j.examples.nlp.glove

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.glove.Glove
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.{Logger, LoggerFactory}

import java.util

/**
  * @author raver119@gmail.com
  */
object GloVeExample {
  private val log: Logger = LoggerFactory.getLogger(GloVeExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    val inputFile = new ClassPathResource("raw_sentences.txt").getFile

    // creating SentenceIterator wrapping our training corpus
    val iter = new BasicLineIterator(inputFile.getAbsolutePath)

    // Split on white spaces in the line to get words
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)

    val glove = new Glove.Builder()
      .iterate(iter)
      .tokenizerFactory(t)
      .alpha(0.75)
      .learningRate(0.1)

      // number of epochs for training
      .epochs(25)

      // cutoff for weighting function
      .xMax(100)

      // training is done in batches taken from training corpus
      .batchSize(1000)

      // if set to true, batches will be shuffled before training
      .shuffle(true)

      // if set to true word pairs will be built in both directions, LTR and RTL
      .symmetric(true)
      .build

    glove.fit()

    val simD = glove.similarity("day", "night")
    log.info("Day/night similarity: " + simD)

    val words: util.Collection[String] = glove.wordsNearest("day", 10)
    log.info("Nearest words to 'day': " + words)

    System.exit(0)
  }
}
