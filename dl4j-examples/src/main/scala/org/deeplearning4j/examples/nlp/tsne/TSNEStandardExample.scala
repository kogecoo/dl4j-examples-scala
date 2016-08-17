package org.deeplearning4j.examples.nlp.tsne

import java.io.File

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.plot.BarnesHutTsne
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable


object TSNEStandardExample {

    lazy val log = LoggerFactory.getLogger(TSNEStandardExample.getClass)

    def main(args: Array[String]) = {
        val iterations = 100
        Nd4j.dtype = DataBuffer.Type.DOUBLE
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE)
        val cacheListBuilder = mutable.ArrayBuilder.make[String]


        log.info("Load & Vectorize data....")
        val wordFile: File = new ClassPathResource("words.txt").getFile
        val vectors  = WordVectorSerializer.loadTxt(wordFile)
        val cache = vectors.getSecond
        val weights: INDArray = vectors.getFirst.getSyn0

        (0 until cache.numWords).foreach { i => cacheListBuilder += cache.wordAtIndex(i) }
        val cacheList = cacheListBuilder.result.toList.asJava

        log.info("Build model....")
        val tsne: BarnesHutTsne = new BarnesHutTsne.Builder()
                .setMaxIter(iterations).theta(0.5)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
                .usePca(false)
                .build()

        log.info("Store TSNE Coordinates for Plotting....")
        val outputFile = "target/archive-tmp/tsne-standard-coords.csv"
        new File(outputFile).getParentFile.mkdirs()
        tsne.plot(weights,2,cacheList,outputFile)
    }



}
