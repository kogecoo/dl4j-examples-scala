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
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.LoggerFactory

import java.io._
import java.util.Random

/**
  * This code example is featured in this youtube video
  *
  * https://www.youtube.com/watch?v=zrTSs715Ylo
  *
  * * This differs slightly from the Video Example,
  * The Video example had the data already downloaded
  * This example includes code that downloads the data
  *
  * Data is downloaded from
  *
  *
  * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
  * followed by tar xzvf mnist_png.tar.gz
  *
  *
  *
  * This examples builds on the MnistImagePipelineExample
  * by Loading the previously saved Neural Net
  */
object MnistImagePipelineExampleLoad {

  /** Data URL for downloading */
  val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

  /** Location to save and extract the training/testing data */
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/")

  private val log = LoggerFactory.getLogger(MnistImagePipelineExampleLoad.getClass)

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
    val numEpochs = 15

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


    log.info("******LOAD TRAINED MODEL******")
    // Details

    // Where the saved model would be if
    // MnistImagePipelineSave has been run
    val locationToSave = new File("trained_mnist_model.zip")

    if (locationToSave.exists) {
      println("\n######Saved Model Found######\n")
    } else {
      println("\n\n#######File not found!#######")
      println("This example depends on running ")
      println("MnistImagePipelineExampleSave")
      println("Run that Example First")
      println("#############################\n\n")


      System.exit(0)
    }




    val model = ModelSerializer.restoreMultiLayerNetwork(locationToSave)


    model.getLabels


    //Test the Loaded Model with the test data

    recordReader.initialize(test)
    val testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)
    scaler.fit(testIter)
    testIter.setPreProcessor(scaler)

    // Create Eval object with 10 possible classes
    val eval = new Evaluation(outputNum)



    while (testIter.hasNext) {
      val next = testIter.next
      val output: INDArray = model.output(next.getFeatures)
      eval.eval(next.getLabels, output)
    }

    log.info(eval.stats)
  }

  @throws[Exception]
  private def downloadData() {
    //Create directory if required
    val directory: File = new File(DATA_PATH)
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
      val tais = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(filePath))))
      try {
        var entry: TarArchiveEntry = tais.getNextEntry.asInstanceOf[TarArchiveEntry]
        /** Read the tar entries using the getNextEntry method **/
        while (entry != null) {
          //System.out.println("Extracting file: " + entry.getName());
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
          if (fileCount % 1000 == 0) System.out.print(".")
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

    if (tmpDirStr == null) {
      throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir")
    }
    val url: String = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
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
