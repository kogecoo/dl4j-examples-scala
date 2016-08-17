package org.deeplearning4j.examples.nlp.paragraphvectors.tools

import java.util.concurrent.atomic.AtomicInteger
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.word2vec.VocabWord
import org.deeplearning4j.text.documentiterator.LabelledDocument
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import javax.validation.constraints.NotNull

import scala.collection.JavaConverters._

/**
 * Simple utility class that builds centroid vector for LabelledDocument
 * based on previously trained ParagraphVectors model
 *
 * @author raver119@gmail.com
 */
class MeansBuilder(@NotNull lookupTable: InMemoryLookupTable[VocabWord],
                   @NotNull tokenizerFactory: TokenizerFactory) {

    private val vocabCache = lookupTable.getVocab

    /**
     * This method returns centroid (mean vector) for document.
     *
     * @param document
     * @return
     */
    def documentAsVector(document: LabelledDocument): INDArray = {
        val documentAsTokens: java.util.List[String]  = tokenizerFactory.create(document.getContent).getTokens
        val cnt: AtomicInteger = new AtomicInteger(0)
        documentAsTokens.asScala.foreach { word =>
            if (vocabCache.containsWord(word)) cnt.incrementAndGet()
        }
        val allWords: INDArray = Nd4j.create(cnt.get(), lookupTable.layerSize())

        cnt.set(0)
        documentAsTokens.asScala.foreach { word =>
            if (vocabCache.containsWord(word))
                allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word))
        }

        allWords.mean(0)
    }
}
