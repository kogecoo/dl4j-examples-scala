package org.deeplearning4j.examples.nlp.paragraphvectors.tools

import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.JavaConverters._
import scala.collection.mutable


/**
 * This is primitive seeker for nearest labels.
 * It's used instead of basic wordsNearest method because for ParagraphVectors
 * only labels should be taken into account, not individual words
 *
 * @author raver119@gmail.com
 */
class LabelSeeker(labelsUsed: java.util.List[String], lookupTable: InMemoryLookupTable[VocabWord]) {

    if (labelsUsed.isEmpty) throw new IllegalStateException("You can't have 0 labels used for ParagraphVectors")

    /**
     * This method accepts vector, that represents any document,
     * and returns distances between this document, and previously trained categories
     * @return
     */
    def getScores(vector: INDArray): java.util.List[Pair[String, Double]] = {
        val builder = mutable.ArrayBuilder.make[Pair[String, Double]]

        labelsUsed.asScala.foreach { label =>
            val vecLabel = lookupTable.vector(label)
            if (vecLabel == null) throw new IllegalStateException("Label '"+ label+"' has no known vector!")

            val sim = Transforms.cosineSim(vector, vecLabel)
            builder += new Pair[String, Double](label, sim)
        }
        builder.result().toList.asJava
    }
}
