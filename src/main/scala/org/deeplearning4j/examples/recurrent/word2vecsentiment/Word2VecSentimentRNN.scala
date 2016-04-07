package org.deeplearning4j.examples.recurrent.word2vecsentiment

import java.io._
import java.net.URL

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.datasets.iterator.{AsyncDataSetIterator, DataSetIterator}
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

/**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 *
 * Process:
 * 1. Download data (movie reviews) + extract. Download + extraction is done automatically.
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network for multiple epochs. At each epoch: evaluate performance on the test set.
 *
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB before extraction.
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file, and extract to a suitable location
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
object Word2VecSentimentRNN {

    /** Data URL for downloading */
    val DATA_URL: String = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    /** Location to save and extract the training/testing data */
    val DATA_PATH: String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/")
    /** Location (local file system) for the Google News vectors. Set this manually. */
    val WORD_VECTORS_PATH: String = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin"


    @throws[IOException]
    def main(args: Array[String]): Unit = {
        //Download and extract data
        downloadData()

        val batchSize = 50     //Number of examples in each minibatch
        val vectorSize = 300   //Size of the word vectors. 300 in the Google News model
        val nEpochs = 5        //Number of epochs (full passes of training data) to train on
        val truncateReviewsToLength = 300  //Truncate reviews with length (# words) greater than this

        //Set up network configuration
        val conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                        .activation("softsign").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
                .pretrain(false).backprop(true).build()

        val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
        net.init()
        net.setListeners(new ScoreIterationListener(1))

        //DataSetIterators for training and testing respectively
        //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
        val wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false)
        val train: DataSetIterator = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,batchSize,truncateReviewsToLength,true),1)
        val test: DataSetIterator = new AsyncDataSetIterator(new SentimentExampleIterator(DATA_PATH,wordVectors,100,truncateReviewsToLength,false),1)

        System.out.println("Starting training")
        (0 until nEpochs).foreach { i =>
            net.fit(train)
            train.reset()
            System.out.println("Epoch " + i + " complete. Starting evaluation:")

            //Run evaluation. This is on 25k reviews, so can take some time
            val evaluation = new Evaluation()
            while(test.hasNext){
                val t: DataSet = test.next()
                val features = t.getFeatureMatrix
                val lables = t.getLabels
                val inMask = t.getFeaturesMaskArray
                val outMask = t.getLabelsMaskArray
                val predicted = net.output(features, false, inMask, outMask)

                evaluation.evalTimeSeries(lables,predicted,outMask)
            }
            test.reset()

            System.out.println(evaluation.stats())
        }


        System.out.println("----- Example complete -----")
    }

    @throws[Exception]
    def downloadData():Unit = {
        //Create directory if required
        val directory = new File(DATA_PATH)
        if(!directory.exists()) directory.mkdir()

        //Download file:
        val archizePath = DATA_PATH + "aclImdb_v1.tar.gz"
        val archiveFile = new File(archizePath)

        if (!archiveFile.exists()) {
            System.out.println("Starting data download (80MB)...")
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath())
            //Extract tar.gz file to output directory
            extractTarGz(archizePath, DATA_PATH)
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath())
        }
    }

    private[this] val BUFFER_SIZE = 4096

    @throws[IOException]
    private[this] def extractTarGz(filePath: String, outputPath: String):Unit = {
        var fileCount = 0
        var dirCount = 0

        System.out.print("Extracting files")
        try {
            val tais: TarArchiveInputStream =
                new TarArchiveInputStream(
                  new GzipCompressorInputStream(
                      new BufferedInputStream(
                          new FileInputStream(filePath)
                      )
                  )
                )

            /** Read the tar entries using the getNextEntry method **/

            var entry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
            while (entry != null) {
                //System.out.println("Extracting file: " + entry.getName())

                //Create directories as required
                if (entry.isDirectory) {
                    new File(outputPath + entry.getName()).mkdirs()
                    dirCount += 1
                } else {
                    val data = new Array[Byte](BUFFER_SIZE)

                    val fos: FileOutputStream = new FileOutputStream(outputPath + entry.getName)
                    val dest: BufferedOutputStream = new BufferedOutputStream(fos,BUFFER_SIZE)

                    var count = tais.read(data, 0, BUFFER_SIZE)
                    while (count != -1) {
                        dest.write(data, 0, count)
                        count = tais.read(data, 0, BUFFER_SIZE)
                    }
                    dest.close()
                    fileCount += 1
                }
                if (fileCount % 1000 == 0) System.out.print(".")

                entry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
            }
        }

        System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath)
    }
}
