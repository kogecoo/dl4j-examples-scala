package org.deeplearning4j.examples.dataExamples

import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.api.split.InputSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.MultiImageTransform
import org.datavec.image.transform.ShowImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.io.File
import java.util.Random

/**
 * Created by susaneraly on 6/9/16.
 */
class ImagePipelineExample {

    protected final val log = LoggerFactory.getLogger(classOf[ImagePipelineExample])

    //Images are of format given by allowedExtension -
    protected final val allowedExtensions: Array[String] = BaseImageLoader.ALLOWED_FORMATS

    protected final val seed: Long = 12345

    final val randNumGen: Random = new Random(seed)

    protected val height = 50
    protected val width = 50
    protected val channels = 3
    protected val numExamples = 80
    protected val outputNum = 4

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
        val parentDir: File = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/DataExamples/ImagePipeline/")
        //Files in directories under the parent dir that have "allowed extensions" plit needs a random number generator for reproducibility when splitting the files into train and test
        val filesInDir: FileSplit = new FileSplit(parentDir, allowedExtensions, randNumGen)

        //You do not have to manually specify labels. This class (instantiated as below) will
        //parse the parent dir and use the name of the subdirectories as label/class names
        val labelMaker: ParentPathLabelGenerator = new ParentPathLabelGenerator()
        //The balanced path filter gives you fine tune control of the min/max cases to load for each class
        //Below is a bare bones version. Refer to javadocs for details
        val pathFilter: BalancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker)

        //Split the image files into train and test. Specify the train test split as 80%,20%
        val filesInDirSplit: Array[InputSplit] = filesInDir.sample(pathFilter, 80, 20)
        val trainData: InputSplit = filesInDirSplit(0)
        val testData: InputSplit = filesInDirSplit(1)

        //Specifying a new record reader with the height and width you want the images to be resized to.
        //Note that the images in this example are all of different size
        //They will all be resized to the height and width specified below
        val recordReader: ImageRecordReader = new ImageRecordReader(height,width,channels,labelMaker)

        //Often there is a need to transforming images to artificially increase the size of the dataset
        //DataVec has built in powerful features from OpenCV
        //You can chain transformations as shown below, write your own classes that will say detect a face and crop to size
        /*ImageTransform transform = new MultiImageTransform(randNumGen,
            new CropImageTransform(10), new FlipImageTransform(),
            new ScaleImageTransform(10), new WarpImageTransform(10))
            */

        //You can use the ShowImageTransform to view your images
        //Code below gives you a look before and after, for a side by side comparison
        val transform: ImageTransform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "))

        //Initialize the record reader with the train data and the transform chain
        recordReader.initialize(trainData,transform)
        //convert the record reader to an iterator for training - Refer to other examples for how to use an iterator
        var dataIter: DataSetIterator = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)
        while (dataIter.hasNext) {
            val ds = dataIter.next()
            println(ds)
            try {
                Thread.sleep(3000);                 //1000 milliseconds is one second.
            } catch { case ex: InterruptedException =>
                Thread.currentThread().interrupt()
            }
        }
        recordReader.reset()

        //transform = new MultiImageTransform(randNumGen,new CropImageTransform(50), new ShowImageTransform("Display - after"))
        //recordReader.initialize(trainData,transform)
        recordReader.initialize(trainData)
        dataIter = new RecordReaderDataSetIterator(recordReader, 10, 1, outputNum)
        while (dataIter.hasNext) {
            val ds = dataIter.next()
        }
        recordReader.reset()

    }
}
