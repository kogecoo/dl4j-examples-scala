package org.deeplearning4j.examples.recurrent.word2vecsentiment

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io._
import java.net.URL

/** Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
  * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
  * (using the Word2Vec model) and fed into a recurrent neural network.
  * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
  * This data set contains 25,000 training reviews + 25,000 testing reviews
  *
  * Process:
  * 1. Automatic on first run of example: Download data (movie reviews) + extract
  * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
  * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
  * 4. Train network
  *
  * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
  * additional tuning.
  *
  * NOTE / INSTRUCTIONS:
  * You will have to download the Google News word vector model manually. ~1.5GB
  * The Google News vector model available here: https://code.google.com/p/word2vec/
  * Download the GoogleNews-vectors-negative300.bin.gz file
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
  val WORD_VECTORS_PATH: String = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin.gz"


  @throws[Exception]
  def main(args: Array[String]) {
    if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) {
      throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example")
    }

    //Download and extract data
    downloadData()

    val batchSize = 64                //Number of examples in each minibatch
    val vectorSize = 300              //Size of the word vectors. 300 in the Google News model
    val nEpochs = 1                   //Number of epochs (full passes of training data) to train on
    val truncateReviewsToLength = 256 //Truncate reviews with length (# words) greater than this

    //Set up network configuration
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .updater(Updater.ADAM)
      .adamMeanDecay(0.9)
      .adamVarDecay(0.999)
      .regularization(true)
      .l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .learningRate(2e-2)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build)
      .pretrain(false).backprop(true).build()

    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    //DataSetIterators for training and testing respectively
    val wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH))
    val train = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true)
    val test = new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false)

    println("Starting training")
    for (i <- 0 until nEpochs) {
      net.fit(train)
      train.reset()
      println("Epoch " + i + " complete. Starting evaluation:")

      //Run evaluation. This is on 25k reviews, so can take some time
      val evaluation = new Evaluation
      while (test.hasNext) {
        val t = test.next
        val features = t.getFeatureMatrix
        val lables = t.getLabels
        val inMask = t.getFeaturesMaskArray
        val outMask = t.getLabelsMaskArray
        val predicted = net.output(features, false, inMask, outMask)
        evaluation.evalTimeSeries(lables, predicted, outMask)
      }
      test.reset()

      println(evaluation.stats)
    }

    //After training: load a single example and generate predictions
    val firstPositiveReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/pos/0_10.txt"))
    val firstPositiveReview = FileUtils.readFileToString(firstPositiveReviewFile)

    val features = test.loadFeaturesFromString(firstPositiveReview, truncateReviewsToLength)
    val networkOutput = net.output(features)
    val timeSeriesLength = networkOutput.size(2)
    val probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all, NDArrayIndex.point(timeSeriesLength - 1))

    println("\n\n-------------------------------")
    println("First positive review: \n" + firstPositiveReview)
    println("\n\nProbabilities at last time step:")
    println("p(positive): " + probabilitiesAtLastWord.getDouble(0))
    println("p(negative): " + probabilitiesAtLastWord.getDouble(1))

    println("----- Example complete -----")
  }

  @throws[Exception]
  private def downloadData() {
    //Create directory if required
    val directory = new File(DATA_PATH)
    if (!directory.exists) directory.mkdir

    //Download file:
    val archizePath = DATA_PATH + "aclImdb_v1.tar.gz"
    val archiveFile = new File(archizePath)
    val extractedPath = DATA_PATH + "aclImdb"
    val extractedFile = new File(extractedPath)

    if (!archiveFile.exists) {
      println("Starting data download (80MB)...")
      FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
      println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath)
      //Extract tar.gz file to output directory
      extractTarGz(archizePath, DATA_PATH)
    } else {
      //Assume if archive (.tar.gz) exists, then data has already been extracted
      println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath)
      if (!extractedFile.exists) {
        //Extract tar.gz file to output directory
        extractTarGz(archizePath, DATA_PATH)
      } else {
        println("Data (extracted) already exists at " + extractedFile.getAbsolutePath)
      }
    }
  }

  private val BUFFER_SIZE = 4096

  @throws[IOException]
  private def extractTarGz(filePath: String, outputPath: String) {
    var fileCount = 0
    var dirCount = 0
    print("Extracting files")
    try {
      val tais: TarArchiveInputStream = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(filePath))))
      try {
        var entry: TarArchiveEntry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
        /** Read the tar entries using the getNextEntry method **/
        while (entry != null) {
          //println("Extracting file: " + entry.getName());
          //Create directories as required
          if (entry.isDirectory) {
            new File(outputPath + entry.getName).mkdirs
            dirCount += 1
          } else {
            val data: Array[Byte] = new Array[Byte](BUFFER_SIZE)
            val fos: FileOutputStream = new FileOutputStream(outputPath + entry.getName)
            val dest: BufferedOutputStream = new BufferedOutputStream(fos, BUFFER_SIZE)

            var count = tais.read(data, 0, BUFFER_SIZE)

            while (count != -1) {
              dest.write(data, 0, count)
              count = tais.read(data, 0, BUFFER_SIZE)
            }
            dest.close()
            fileCount += 1
          }
          if (fileCount % 1000 == 0) print(".")

          entry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
        }
      } finally {
        if (tais != null) tais.close()
      }
    }
    println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath)
  }
}
