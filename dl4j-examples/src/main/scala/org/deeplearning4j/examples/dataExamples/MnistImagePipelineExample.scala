package org.deeplearning4j.examples.dataExamples

import org.apache.commons.compress.archivers.tar.{TarArchiveEntry, TarArchiveInputStream}
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.FilenameUtils
import org.apache.http.HttpEntity
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.{CloseableHttpClient, HttpClientBuilder}
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.records.listener.impl.LogRecordListener
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.{Logger, LoggerFactory}

import java.io._
import java.util.Random

/**
  * Created by tom hanlon on 11/7/16.
  * This code example is featured in this youtube video
  * https://www.youtube.com/watch?v=GLC8CIoHDnI
  *
  * This differs slightly from the Video Example,
  * The Video example had the data already downloaded
  * This example includes code that downloads the data
  *
  * Instructions
  * Downloads a directory containing a testing and a training folder
  * each folder has 10 directories 0-9
  * in each directory are 28 * 28 grayscale pngs of handwritten digits
  * The training and testing directories will have directories 0-9 with
  * 28 * 28 PNG images of handwritten images
  *
  * The code here shows how to use a ParentPathLabelGenerator to label the images as
  * they are read into the RecordReader
  *
  * The pixel values are scaled to values between 0 and 1 using
  * ImagePreProcessingScaler
  *
  * In this example a loop steps through 3 images and prints the DataSet to
  * the terminal. The expected output is the 28* 28 matrix of scaled pixel values
  * the list with the label for that image
  * and a list of the label values
  *
  * This example also applies a Listener to the RecordReader that logs the path of each image read
  * You would not want to do this in production
  * The reason it is done here is to show that a handwritten image 3 (for example)
  * was read from directory 3,
  * has a matrix with the shown values
  * Has a label value corresponding to 3
  *
  */
object MnistImagePipelineExample {

  /** Data URL for downloading */
  val DATA_URL = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"

  /** Location to save and extract the training/testing data */
  val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_Mnist/")



  private val log: Logger = LoggerFactory.getLogger(MnistImagePipelineExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    /*
    image information
    28 * 28 grayscale
    grayscale implies single channel
    */
    val height = 28
    val width = 28
    val channels = 1
    val rngseed = 123
    val randNumGen = new Random(rngseed)
    val batchSize = 1
    val outputNum = 10


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

    // The LogRecordListener will log the path of each image read
    // used here for information purposes,
    // If the whole dataset was ingested this would place 60,000
    // lines in our logs
    // It will show up in the output with this format
    // o.d.a.r.l.i.LogRecordListener - Reading /tmp/mnist_png/training/4/36384.png

    recordReader.setListeners(new LogRecordListener)

    // DataSet Iterator

    val dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum)

    // Scale pixel values to 0-1

    val scaler = new ImagePreProcessingScaler(0, 1)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)

    // In production you would loop through all the data
    // in this example the loop is just through 3
    // images for demonstration purposes
    for (i <- 1 until 3) {
      val ds = dataIter.next()
      println(ds)
      println(dataIter.getLabels)
    }
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
    var fileCount: Int = 0
    var dirCount: Int = 0
    print("Extracting files")
    try {
      val tais = new TarArchiveInputStream(new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(filePath))))
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
            val fos= new FileOutputStream(outputPath + entry.getName)
            val dest= new BufferedOutputStream(fos, BUFFER_SIZE)

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
    if (Option(tmpDirStr).isEmpty) {
      throw new IOException("System property 'java.io.tmpdir' does specify a tmp dir")
    }
    val url = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
    val f = new File(archizePath)
    val dir = new File(tmpDirStr)
    if (!f.exists) {
      val builder: HttpClientBuilder = HttpClientBuilder.create
      val client: CloseableHttpClient = builder.build
      try {
        val response = client.execute(new HttpGet(url))
        try {
          val entity: HttpEntity = response.getEntity
          if (Option(entity).isDefined) {
            try {
              val outstream: FileOutputStream = new FileOutputStream(f)
              try {
                entity.writeTo(outstream)
                outstream.flush()
                outstream.close()
              } finally {
                if (Option(outstream).isDefined) outstream.close()
              }
            }
          }
        } finally {
          if (Option(response).isDefined) response.close()
        }
      }
      println("Data downloaded to " + f.getAbsolutePath)
    }
    else {
      println("Using existing directory at " + f.getAbsolutePath)
    }
  }
}
