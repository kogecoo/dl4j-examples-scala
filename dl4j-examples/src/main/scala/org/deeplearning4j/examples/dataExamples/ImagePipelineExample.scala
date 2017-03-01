package org.deeplearning4j.examples.dataExamples

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{ImageTransform, MultiImageTransform, ShowImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.util.Random

/**
  * Created by susaneraly on 6/9/16.
  */
object ImagePipelineExample {

  protected val log: Logger = LoggerFactory.getLogger(ImagePipelineExample.getClass)

  //Images are of format given by allowedExtension -
  protected val allowedExtensions: Array[String] = BaseImageLoader.ALLOWED_FORMATS

  protected val seed: Long = 12345

  val randNumGen = new Random(seed)

  protected var height = 50
  protected var width = 50
  protected var channels = 3
  protected var numExamples = 80
  protected var outputNum = 4

  @throws[Exception]
  def main(args: Array[String]) {
    //DIRECTORY STRUCTURE:
    //Images in the dataset have to be organized in directories by class/label.
    //In this example there are ten images in three classes
    //Here is the directory structure
    //                                    parentDir
    //                                  /    |     \
    //                                 /     |      \
    //                            labelA  labelB   labelC
    //
    //Set your data up like this so that labels from each label/class live in their own directory
    //And these label/class directories live together in the parent directory
    //
    //
    val parentDir = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/")
    //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into train and test
    val filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen)

    //You do not have to manually specify labels. This class (instantiated as below) will
    //parse the parent dir and use the name of the subdirectories as label/class names
    val labelMaker = new ParentPathLabelGenerator
    //The balanced path filter gives you fine tune control of the min/max cases to load for each class
    //Below is a bare bones version. Refer to javadocs for details
    val pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)

    //Split the image files into train and test. Specify the train test split as 80%,20%
    val filesInDirSplit = filesInDir.sample(pathFilter, 80, 20)
    val trainData = filesInDirSplit(0)
    val testData = filesInDirSplit(1)

    //Specifying a new record reader with the height and width you want the images to be resized to.
    //Note that the images in this example are all of different size
    //They will all be resized to the height and width specified below
    val recordReader = new ImageRecordReader(height, width, channels, labelMaker)

    //Often there is a need to transforming images to artificially increase the size of the dataset
    //DataVec has built in powerful features from OpenCV
    //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
    /*ImageTransform transform = new MultiImageTransform(randNumGen,
    new CropImageTransform(10), new FlipImageTransform(),
    new ScaleImageTransform(10), new WarpImageTransform(10));
    */
    //You can use the ShowImageTransform to view your images
    //Code below gives you a look before and after, for a side by side comparison
    val transform: ImageTransform = new MultiImageTransform(randNumGen, new ShowImageTransform("Display - before "))

    //Initialize the record reader with the train data and the transform chain
    recordReader.initialize(trainData, transform)
    //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
    var dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)

    while (dataIter.hasNext) {
      val ds = dataIter.next()
      println(ds)
      try {
        Thread.sleep(3000) //1000 milliseconds is one second.
      } catch { case ex: InterruptedException => Thread.currentThread.interrupt() }
    }
    recordReader.reset()

    //transform = new MultiImageTransform(randNumGen,new CropImageTransform(50), new ShowImageTransform("Display - after"));
    //recordReader.initialize(trainData,transform);
    recordReader.initialize(trainData)
    dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)
    while (dataIter.hasNext) {
      val ds = dataIter.next()
    }
    recordReader.reset()
  }
}
