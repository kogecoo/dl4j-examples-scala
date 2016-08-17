package org.deeplearning4j.examples.nlp.paragraphvectors

import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import org.deeplearning4j.text.documentiterator.LabelsSource
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, SentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of all sentences present in training corpus.
 * However, you still use it for training on labelled documents, using sets of LabelledDocument and LabelAwareIterator implementation.
 *
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
object ParagraphVectorsTextExample {

    lazy val log = LoggerFactory.getLogger(ParagraphVectorsTextExample.getClass)

    @throws[Exception]
    def main(args: Array[String]):Unit = {
        val resource = new ClassPathResource("/raw_sentences.txt")
        val file = resource.getFile
        val iter: SentenceIterator  = new BasicLineIterator(file)

        val cache: InMemoryLookupCache = new InMemoryLookupCache()

        val t: TokenizerFactory  = new DefaultTokenizerFactory()
        t.setTokenPreProcessor(new CommonPreprocessor())

        /*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.

              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        */
        val source: LabelsSource = new LabelsSource("DOC_")

        val vec: ParagraphVectors = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(5)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build()

        vec.fit()

        /*
            In training corpus we have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space

            line 3721: This is my way .
            line 6348: This is my case .
            line 9836: This is my house .
            line 12493: This is my world .
            line 16393: This is my work .

            this is special sentence, that has nothing common with previous sentences
            line 9853: We now have one .

            Note that docs are indexed from 0
         */

        val similarity1: Double = vec.similarity("DOC_9835", "DOC_12492")
        log.info("9836/12493 ('This is my house .'/'This is my world .') similarity: " + similarity1)

        val similarity2: Double = vec.similarity("DOC_3720", "DOC_16392")
        log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2)

        val similarity3: Double = vec.similarity("DOC_6347", "DOC_3720")
        log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3)

        // likelihood in this case should be significantly lower
        val similarityX: Double = vec.similarity("DOC_3720", "DOC_9852")
        log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
          "(should be significantly lower)")
    }
}
