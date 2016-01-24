package org.deeplearning4j.examples.word2vec

import org.canova.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.{VocabWord, Word2Vec}
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import org.deeplearning4j.text.sentenceiterator.{BasicLineIterator, SentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.slf4j.LoggerFactory

/**
 * This is simple example for model weights update after initial vocab building.
 * If you have built your w2v model, and some time later you've decided that it can be additionally trained over new corpus, here's an example how to do it.
 *
 * PLEASE NOTE: At this moment, no new words will be added to vocabulary/model. Only weights update process will be issued. It's often called "frozen vocab training".
 *
 * @author raver119@gmail.com
 */
object Word2VecUptrainingExample {

    lazy val log = LoggerFactory.getLogger(Word2VecRawTextExample.getClass)

    def main(args: Array[String]): Unit = {
        /*
                Initial model training phase
         */
        val filePath = new ClassPathResource("raw_sentences.txt").getFile.getAbsolutePath

        log.info("Load & Vectorize Sentences....")
        // Strip white space before and after for each line
        val iter = new BasicLineIterator(filePath)
        // Split on white spaces in the line to get words
        val t = new DefaultTokenizerFactory()
        t.setTokenPreProcessor(new CommonPreprocessor())

        // manual creation of VocabCache and WeightLookupTable usually isn't necessary
        // but in this case we'll need them
        // manual creation of VocabCache and WeightLookupTable usually isn't necessary
        // but in this case we'll need them
        val cache = new InMemoryLookupCache()
        val table = new InMemoryLookupTable.Builder[VocabWord]()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build()


        log.info("Building model....")
        val vec: Word2Vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .epochs(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .lookupTable(table)
                .vocabCache(cache)
                .build()

        log.info("Fitting Word2Vec model....")
        vec.fit()


        val lst1: java.util.Collection[String] = vec.wordsNearest("day", 10)
        log.info("Closest words to 'day' on 1st run: " + lst1)

        /*
            at this momen we're supposed to have model built, and it can be saved for future use.
         */
        WordVectorSerializer.writeFullModel(vec, "pathToSaveModel.txt")

        /*
            Let's assume that some time passed, and now we have new corpus to be used to weights update.
            Instead of building new model over joint corpus, we can use weights update mode.
         */
        val word2Vec: Word2Vec = WordVectorSerializer.loadFullModel("pathToSaveModel.txt")

        /*
            PLEASE NOTE: after model is restored, it's still required to set SentenceIterator and TokenizerFactory, if you're going to train this model
         */
        val iterator: SentenceIterator = new BasicLineIterator(filePath)
        val tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory()
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor())

        word2Vec.setSentenceIter(iterator)
        word2Vec.setTokenizerFactory(tokenizerFactory)

        log.info("Word2vec uptraining...")

        word2Vec.fit()

        val lst2 = word2Vec.wordsNearest("day", 10)
        log.info("Closest words to 'day' on 2nd run: " + lst2)

        /*
            Model can be saved for future use now
         */
    }
}
