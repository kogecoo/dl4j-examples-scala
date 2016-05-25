package org.deeplearning4j.examples.nlp.paragraphvectors

import org.canova.api.util.ClassPathResource
import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.{FileLabelAwareIterator, LabelSeeker, MeansBuilder}
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.text.documentiterator.{LabelAwareIterator, LabelledDocument}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.LoggerFactory

import java.io.FileNotFoundException

import scala.collection.JavaConverters._


/**
 * This is basic example for documents classification done with DL4j ParagraphVectors.
 * The overall idea is to use ParagraphVectors in the same way we use LDA:
 * topic space modelling.
 *
 * In this example we assume we have few labeled categories that we can use
 * for training, and few unlabeled documents. And our goal is to determine,
 * which category these unlabeled documents fall into
 *
 *
 * Please note: This example could be improved by using learning cascade
 * for higher accuracy, but that's beyond basic example paradigm.
 *
 * @author raver119@gmail.com
 */
object ParagraphVectorsClassifierExample {

    val log = LoggerFactory.getLogger(ParagraphVectorsClassifierExample.getClass)


    @throws[Exception]
    def main(args: Array[String]): Unit = {
        val app = new ParagraphVectorsClassifierExample()
        val (paragraphVectors, iterator, tokenizerFactory) = app.makeParagraphVectors()
        app.checkUnlabeledData(paragraphVectors, iterator, tokenizerFactory)
    }
}


class ParagraphVectorsClassifierExample {

  @throws[Exception]
  def makeParagraphVectors(): (ParagraphVectors, LabelAwareIterator, TokenizerFactory) = {
    val resource = new ClassPathResource("paravec/labeled")

    // build a iterator for our dataset
    val iterator = new FileLabelAwareIterator.Builder()
      .addSourceFolder(resource.getFile)
      .build()

    val tokenizerFactory = new DefaultTokenizerFactory()
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor())

    // ParagraphVectors training configuration
    val paragraphVectors = new ParagraphVectors.Builder()
      .learningRate(0.025)
      .minLearningRate(0.001)
      .batchSize(1000)
      .epochs(20)
      .iterate(iterator)
      .trainWordVectors(true)
      .tokenizerFactory(tokenizerFactory)
      .build()

    // Start model training
    paragraphVectors.fit()
    (paragraphVectors, iterator, tokenizerFactory)
  }

  @throws[FileNotFoundException]
  def checkUnlabeledData(paragraphVectors: ParagraphVectors, iterator: LabelAwareIterator, tokenizerFactory: TokenizerFactory ) = {
    /*
      At this point we assume that we have model built and we can check
      which categories our unlabeled document falls into.
      So we'll start loading our unlabeled documents and checking them
     */
    val unClassifiedResource = new ClassPathResource("paravec/unlabeled")
    val unClassifiedIterator = new FileLabelAwareIterator.Builder()
      .addSourceFolder(unClassifiedResource.getFile)
      .build()

    /*
      Now we'll iterate over unlabeled data, and check which label it could be assigned to
      Please note: for many domains it's normal to have 1 document fall into few labels at once,
      with different "weight" for each.
     */
    val meansBuilder = new MeansBuilder(
      paragraphVectors.getLookupTable.asInstanceOf[InMemoryLookupTable[VocabWord]],
    tokenizerFactory)
    val seeker = new LabelSeeker(iterator.getLabelsSource.getLabels,
      paragraphVectors.getLookupTable.asInstanceOf[InMemoryLookupTable[VocabWord]])

    while (unClassifiedIterator.hasNextDocument()) {
      val document: LabelledDocument = unClassifiedIterator.nextDocument()
      val documentAsCentroid: INDArray = meansBuilder.documentAsVector(document)
      val scores: java.util.List[Pair[String, Double]] = seeker.getScores(documentAsCentroid)

      /*
          please note, document.getLabel() is used just to show which document we're looking at now,
          as a substitute for printing out the whole document name.
          So, labels on these two documents are used like titles,
          just to visualize our classification done properly
         */
      ParagraphVectorsClassifierExample.log.info("Document '" + document.getLabel + "' falls into the following categories: ")
      scores.asScala.foreach { score =>
        ParagraphVectorsClassifierExample.log.info("        " + score.getFirst + ": " + score.getSecond)
      }
    }
  }
}
