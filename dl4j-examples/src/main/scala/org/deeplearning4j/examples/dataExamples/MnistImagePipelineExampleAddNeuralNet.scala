package org.deeplearning4j.examples.dataExamples

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FilenameUtils
import org.apache.http.HttpEntity
import org.apache.http.client.methods.{CloseableHttpResponse, HttpGet}
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import java.io._
import java.util.Random

/**
  * This code example is featured in this youtube video
  * https://www.youtube.com/watch?v=ECA6y6ahH5E
  *
  * * This differs slightly from the Video Example,
  * The Video example had the data already downloaded
  * This example includes code that downloads the data
  *
  *
  * The Data Directory mnist_png will have two child directories training and testing
  * The training and testing directories will have directories 0-9 with
  * 28 * 28 PNG images of handwritten images
  *
  *
  *
  * The data is downloaded from
  * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
  * followed by tar xzvf mnist_png.tar.gz
  *
  *
  *
  * This examples builds on the MnistImagePipelineExample
  * by adding a Neural Net
  */
object MnistImagePipelineExampleAddNeuralNet {
  private val log: Logger = LoggerFactory.getLogger(MnistImagePipelineExampleAddNeuralNet.getClass)

  /** Data URL for downloading */
  val DATA_URL: String = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

  /** Location to save and extract the training/testing data */
  val DATA_PATH: String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/")

  @throws[Exception]
  def main(args: Array[String]) {

    // image information
    // 28 * 28 grayscale
    // grayscale implies single channel
    val height = 28
    val width = 28
    val channels = 1
    val rngseed = 123
    val randNumGen = new Random(rngseed)
    val batchSize = 128
    val outputNum = 10
    val numEpochs = 1

    /*
    This class downloadData() downloads the data
    stores the data in java's tmpdir
    15MB download compressed
    It will take 158MB of space when uncompressed
    The data can be downloaded manually here
    http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
    */

    downloadData()

    // Define the File Paths
    val trainData = new File(DATA_PATH + "/mnist_png/training")
    val testData = new File(DATA_PATH + "/mnist_png/testing")

    // Define the File Paths
    //val trainData = new File("/tmp/mnist_png/training");
    //val testData = new File("/tmp/mnist_png/testing");

    // Define the FileSplit(PATH, ALLOWED FORMATS,random)

    val train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
    val test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)

    // Extract the parent path as the image label

    val labelMaker = new ParentPathLabelGenerator

    val recordReader = new ImageRecordReader(height, width, channels, labelMaker)

    // Initialize the record reader
    // add a listener, to extract the name

    recordReader.initialize(train)
    //recordReader.setListeners(new LogRecordListener());

    // DataSet Iterator

    val dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)

    // Scale pixel values to 0-1

    val scaler = new ImagePreProcessingScaler(0, 1)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)


    // Build Our Neural Network

    log.info("**** Build Model ****")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(rngseed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.006)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .regularization(true)
      .l2(1e-4)
      .list
      .layer(0, new DenseLayer.Builder()
        .nIn(height * width)
        .nOut(100)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nIn(100)
        .nOut(outputNum)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build)
      .pretrain(false).backprop(true)
      .setInputType(InputType.convolutional(height, width, channels))
      .build

    val model = new MultiLayerNetwork(conf)

    // The Score iteration Listener will log
    // output to show how well the network is training
    model.setListeners(new ScoreIterationListener(10))

    log.info("*****TRAIN MODEL********")
    for (i <- 0 until numEpochs) {
      model.fit(dataIter)
    }

    log.info("******EVALUATE MODEL******")

    recordReader.reset()

    // The model trained on the training dataset split
    // now that it has trained we evaluate against the
    // test data of images the network has not seen

    recordReader.initialize(test)
    val testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)
    scaler.fit(testIter)
    testIter.setPreProcessor(scaler)

    /*
    log the order of the labels for later use
    In previous versions the label order was consistent, but random
    In current verions label order is lexicographic
    preserving the RecordReader Labels order is no
    longer needed left in for demonstration
    purposes
    */
    log.info(recordReader.getLabels.toString)

    // Create Eval object with 10 possible classes
    val eval = new Evaluation(outputNum)

    // Evaluate the network
    while (testIter.hasNext) {
      val next = testIter.next
      val output = model.output(next.getFeatureMatrix)
      // Compare the Feature Matrix from the model
      // with the labels from the RecordReader
      eval.eval(next.getLabels, output)
    }
    log.info(eval.stats)
  }

  @throws[Exception]
  private def downloadData() {
    //Create directory if required
    val directory = new File(DATA_PATH)
    if (!directory.exists) directory.mkdir

    //Download file:
    val archizePath = DATA_PATH + "/mnist_png.tar.gz"
    val archiveFile = new File(archizePath)
    val extractedPath = DATA_PATH + "mnist_png"
    val extractedFile = new File(extractedPath)

    if (!archiveFile.exists) {
      println("Starting data download (15MB)...")
      getMnistPNG()
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

  private val BUFFER_SIZE: Int = 4096

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
            val data = new Array[Byte](BUFFER_SIZE)
            val fos = new FileOutputStream(outputPath + entry.getName)
            val dest = new BufferedOutputStream(fos, BUFFER_SIZE)
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

  @throws[IOException]
  def getMnistPNG() {
    val tmpDirStr = System.getProperty("java.io.tmpdir")
    val archizePath = DATA_PATH + "/mnist_png.tar.gz"
    if (Option(tmpDirStr).isDefined) {
      throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir")
    }
    val url = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
    val f = new File(archizePath)
    val dir = new File(tmpDirStr)
    if (!f.exists) {
      val builder: HttpClientBuilder = HttpClientBuilder.create
      val client: CloseableHttpClient = builder.build
      try {
        val response: CloseableHttpResponse = client.execute(new HttpGet(url))
        try {
          val entity: HttpEntity = response.getEntity
          if (entity != null) {
            try {
              val outstream: FileOutputStream = new FileOutputStream(f)
              try {
                entity.writeTo(outstream)
                outstream.flush()
                outstream.close()
              } finally {
                if (outstream != null) outstream.close()
              }
            }
          }
        } finally {
          if (response != null) response.close()
        }
      }
      println("Data downloaded to " + f.getAbsolutePath)
    } else {
      println("Using existing directory at " + f.getAbsolutePath)
    }
  }
}
