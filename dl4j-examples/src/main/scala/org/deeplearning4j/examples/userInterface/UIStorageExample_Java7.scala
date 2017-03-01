package org.deeplearning4j.examples.userInterface

import java.io.File

import org.deeplearning4j.examples.userInterface.util.UIExampleUtils
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.stats.J7StatsListener
import org.deeplearning4j.ui.storage.sqlite.J7FileStatsStorage

/**
  * A variant of the UI example showing the approach for Java 7 compatibility
  *
  * *** Notes ***
  * 1: If you don't specifically need Java 7, use the approach in the standard UIStorageExample as it should be faster
  * 2: The UI itself requires Java 8 (uses the Play framework as a backend). But you can store stats on one machine, copy
  * the file to another (with Java 8) and visualize there
  * 3: J7FileStatsStorage and FileStatsStorage formats are NOT compatible. Save/load with the same one
  * (J7FileStatsStorage works on Java 8 too, but FileStatsStorage does not work on Java 7)
  *
  * @author Alex Black
  */
object UIStorageExample_Java7 {

  def main(args: Array[String]) {
    //Run this example twice - once with collectStats = true, and then again with collectStats = false
    val collectStats = true

    val statsFile = new File("UIStorageExampleStats_Java7.dl4j")

    //First run training stats from the network
    //Note that we don't have to actually plot it when we collect it - though we can do that too, if required

    val net = UIExampleUtils.getMnistNetwork
    val trainData = UIExampleUtils.getMnistData

    val statsStorage = new J7FileStatsStorage(statsFile) //Note the J7
    net.setListeners(new J7StatsListener(statsStorage), new ScoreIterationListener(10))

    net.fit(trainData)

    println("Done")
  }
}
