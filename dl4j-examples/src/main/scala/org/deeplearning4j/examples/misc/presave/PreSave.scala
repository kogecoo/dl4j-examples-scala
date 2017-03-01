package org.deeplearning4j.examples.misc.presave

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.LoggerFactory

import java.io.File

/**
  * Pre saving the dataset is crucial.
  * Unlike with other frameworks that force you
  * to use 1 data format, deeplearning4j,
  * allows you to load arbitrary data, and also provides
  * tools such as datavec for pre processing a wide variety
  * of data from text, images, video, to log data.
  *
  * In this example, we pre show how to use a datasetiterator
  * to save pre save data.
  * In the other class {@link LoadPreSavedLenetMnistExample}
  * we then use the output to load data from the trainFolder
  * and testFolder.
  *
  * By pre saving the datasets, we save ALOT of time.
  * Anytime you end up trying to re do the processing every time
  * it ends up being a bottleneck.
  *
  * Pre saving the data allows you to have higher throughput during training.
  *
  * @author Adam Gibson
  */
object PreSave {
  private val log = LoggerFactory.getLogger(LoadPreSavedLenetMnistExample.getClass)

  @throws[Exception]
  def main(args: Array[String]) {
    val batchSize: Int = 64 // Test batch size
    /*
       Create an iterator using the batch size for one iteration
    */
    log.info("Load data....")
    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, 12345)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, 12345)
    val trainFolder: File = new File("trainFolder")
    trainFolder.mkdirs
    val testFolder: File = new File("testFolder")
    testFolder.mkdirs
    log.info("Saving train data to " + trainFolder.getAbsolutePath + " and test data to " + testFolder.getAbsolutePath)
    //Track the indexes of the files being saved.
    //These batch indexes are used for indexing which minibatch is being saved by the iterator.
    var trainDataSaved: Int = 0
    var testDataSaved: Int = 0
    while (mnistTrain.hasNext) {
      //note that we use testDataSaved as an index in to which batch this is for the file
      mnistTrain.next().save(new File(trainFolder, "mnist-train-" + trainDataSaved + ".bin"))
      //^^^^^^^
      //******************
      //YOU NEED TO KNOW WHAT THIS IS.
      //This is the index for the file saved.
      //******************************************
      trainDataSaved += 1
    }

    while (mnistTest.hasNext) {
      //note that we use testDataSaved as an index in to which batch this is for the file
      mnistTest.next().save(new File(testFolder, "mnist-test-" + testDataSaved + ".bin"))
      //^^^^^^^
      //******************
      //YOU NEED TO KNOW WHAT THIS IS.
      //This is the index for the file saved.
      //******************************************
      testDataSaved += 1
    }

    log.info("Finished pre saving test and train data")
  }
}
