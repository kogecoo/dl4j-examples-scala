package org.deeplearning4j.examples.nlp.paragraphvectors.tools

import lombok.NonNull
import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

import java.util

import scala.collection.JavaConverters._

/**
  * This is primitive seeker for nearest labels.
  * It's used instead of basic wordsNearest method because for ParagraphVectors
  * only labels should be taken into account, not individual words
  *
  * @author raver119@gmail.com
  */
class LabelSeeker(var labelsUsed: util.List[String], var lookupTable: InMemoryLookupTable[VocabWord]) {
  if (labelsUsed.isEmpty)
    throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors")

  /**
    * This method accepts vector, that represents any document,
    * and returns distances between this document, and previously trained categories
    *
    * @return
    */
  def getScores(@NonNull vector: INDArray): util.List[Pair[String, Double]] = {
    val result = new util.ArrayList[Pair[String, Double]]

    for (label <- labelsUsed.asScala) {
      val vecLabel: INDArray = lookupTable.vector(label)

      if (vecLabel == null)
        throw new IllegalStateException("Label '" + label + "' has no known vector!")

      val sim = Transforms.cosineSim(vector, vecLabel)
      result.add(new Pair[String, Double](label, sim))
    }
    result
  }
}
