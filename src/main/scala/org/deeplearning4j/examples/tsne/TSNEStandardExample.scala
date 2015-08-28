package org.deeplearning4j.examples.tsne

import org.deeplearning4j.berkeley.Pair
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.wordstore.VocabCache
import org.deeplearning4j.plot.Tsne
import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource

import java.io.File
import java.util.{ ArrayList, List }
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuilder

object TSNEStandardExample {

    lazy val log = LoggerFactory.getLogger(TSNEStandardExample.getClass)

    def main(args: Array[String]) = {
        val iterations = 1000
        //List<String> cacheList = new ArrayList<>()
        val cacheListBuilder = ArrayBuilder.make[String]


        log.info("Load & Vectorize data....")
        val wordFile: File = new ClassPathResource("words.txt").getFile()
        val vectors: Pair[InMemoryLookupTable, VocabCache]  = WordVectorSerializer.loadTxt(wordFile)
        val cache: VocabCache = vectors.getSecond()
        val weights: INDArray = vectors.getFirst().getSyn0()

        (0 until cache.numWords).foreach { i => cacheListBuilder += cache.wordAtIndex(i) }
        val cacheList = cacheListBuilder.result.toList.asJava

        log.info("Build model....")
        val tsne = new Tsne.Builder()
                .setMaxIter(iterations)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
                .usePca(false)
                .build()

        log.info("Store TSNE Coordinates for Plotting....")
        val outputFile = "target/archive-tmp/tsne-standard-coords.csv"
        (new File(outputFile)).getParentFile().mkdirs()
        tsne.plot(weights,2,cacheList,outputFile)
    }



}
