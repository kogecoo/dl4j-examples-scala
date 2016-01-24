package org.deeplearning4j.examples.word2vec

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import org.deeplearning4j.text.sentenceiterator.{SentenceIterator, UimaSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource

object Word2VecRawTextExample {

    lazy val log = LoggerFactory.getLogger(Word2VecRawTextExample.getClass)

    def main(args: Array[String]) = {

        val filePath = new ClassPathResource("raw_sentences.txt").getFile.getAbsolutePath

        log.info("Load & Vectorize Sentences....")
        // Strip white space before and after for each line
        val iter: SentenceIterator = UimaSentenceIterator.createWithPath(filePath)
        // Split on white spaces in the line to get words
        val t = new DefaultTokenizerFactory()
        t.setTokenPreProcessor(new CommonPreprocessor())

        val cache = new InMemoryLookupCache()
        val table = new InMemoryLookupTable.Builder()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build()

        log.info("Building model....")
        val vec = new Word2Vec.Builder()
                .minWordFrequency(5).iterations(1)
                .layerSize(100).lookupTable(table)
                .stopWords(new java.util.ArrayList[String]())
                .vocabCache(cache).seed(42)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build()

        log.info("Fitting Word2Vec model....")
        vec.fit()

        log.info("Writing word vectors to text file....")
        // Write word
        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt")

        log.info("Closest Words:")
        val lst: java.util.Collection[String] = vec.wordsNearest("day", 10)
        System.out.println(lst)
    }
}
