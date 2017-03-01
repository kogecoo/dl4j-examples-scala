package org.deeplearning4j.examples.dataExamples

import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import javax.swing._

/**
  *
  * This code example is featured in this youtube video
  *
  * http://www.youtube.com/watch?v=DRHIpeJpJDI
  *
  * This differs slightly from the Video Example,
  * The Video example had the data already downloaded
  * This example includes code that downloads the data
  *
  * Data is downloaded from
  *
  *
  * wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
  * followed by tar xzvf mnist_png.tar.gz
  * The Data Directory mnist_png will have two child directories training and testing
  * The training and testing directories will have directories 0-9 with
  * 28 * 28 PNG images of handwritten images
  *
  *
  *
  *
  *
  * This examples builds on the MnistImagePipelineExample
  * by giving the user a file chooser to test an image of their choice
  * against the Nueral Net, will the network think this cat is an 8 or a 1
  * Seriously you can test anything, but obviously the network was trained on handwritten images
  * 0-9 white digit, black background, so it will work better with stuff closer to what it was
  * designed for
  *
  */
object MnistImagePipelineLoadChooser {
  private val log: Logger = LoggerFactory.getLogger(MnistImagePipelineLoadChooser.getClass)

  /*
  Create a popup window to allow you to chose an image file to test against the
  trained Neural Network
  Chosen images will be automatically
  scaled to 28*28 grayscale
  */
  def fileChose: String = {
    val fc: JFileChooser = new JFileChooser
    val ret: Int = fc.showOpenDialog(null)
    if (ret == JFileChooser.APPROVE_OPTION) {
      val file: File = fc.getSelectedFile
      val filename: String = file.getAbsolutePath
      filename
    } else null
  }

  @throws[Exception]
  def main(args: Array[String]) {
    val height = 28
    val width = 28
    val channels = 1

    // recordReader.getLabels()
    // In this version Labels are always in order
    // So this is no longer needed
    //List<Integer> labelList = Arrays.asList(2,3,7,1,6,4,0,5,8,9);
    val labelList = Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    // pop up file chooser
    val filechose = fileChose.toString

    //LOAD NEURAL NETWORK

    // Where to save model
    val locationToSave: File = new File("trained_mnist_model.zip")
    // Check for presence of saved model
    if (locationToSave.exists) {
      println("\n######Saved Model Found######\n")
    }
    else {
      println("\n\n#######File not found!#######")
      println("This example depends on running ")
      println("MnistImagePipelineExampleSave")
      println("Run that Example First")
      println("#############################\n\n")

      System.exit(0)
    }

    val model = ModelSerializer.restoreMultiLayerNetwork(locationToSave)

    log.info("*********TEST YOUR IMAGE AGAINST SAVED NETWORK********")

    // FileChose is a string we will need a file

    val file = new File(filechose)

    // Use NativeImageLoader to convert to numerical matrix

    val loader = new NativeImageLoader(height, width, channels)

    // Get the image into an INDarray

    val image = loader.asMatrix(file)

    // 0-255
    // 0-1
    val scaler = new ImagePreProcessingScaler(0, 1)
    scaler.transform(image)
    // Pass through to neural Net

    val output = model.output(image)

    log.info("## The FILE CHOSEN WAS " + filechose)
    log.info("## The Neural Nets Pediction ##")
    log.info("## list of probabilities per label ##")
    //log.info("## List of Labels in Order## ");
    // In new versions labels are always in order
    log.info(output.toString)
    log.info(labelList.toString)
  }

}
