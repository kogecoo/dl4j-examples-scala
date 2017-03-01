package org.deeplearning4j.examples.nlp.paragraphvectors.tools

import lombok.NonNull
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.text.documentiterator.LabelledDocument
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._

/**
  * Simple utility class that builds centroid vector for LabelledDocument
  * based on previously trained ParagraphVectors model
  *
  * @author raver119@gmail.com
  */
class MeansBuilder(var lookupTable: InMemoryLookupTable[VocabWord], var tokenizerFactory: TokenizerFactory) {

  private var vocabCache = lookupTable.getVocab

  /**
    * This method returns centroid (mean vector) for document.
    *
    * @param document
    * @return
    */
  def documentAsVector(@NonNull document: LabelledDocument): INDArray = {
    val documentAsTokens = tokenizerFactory.create(document.getContent).getTokens
    val cnt = new AtomicInteger(0)

    for (word <- documentAsTokens.asScala) {
      if (vocabCache.containsWord(word)) cnt.incrementAndGet
    }

    val allWords: INDArray = Nd4j.create(cnt.get, lookupTable.layerSize)
    cnt.set(0)
    for (word <- documentAsTokens.asScala) {
      if (vocabCache.containsWord(word)) allWords.putRow(cnt.getAndIncrement, lookupTable.vector(word))
    }

    allWords.mean(0)
  }
}
