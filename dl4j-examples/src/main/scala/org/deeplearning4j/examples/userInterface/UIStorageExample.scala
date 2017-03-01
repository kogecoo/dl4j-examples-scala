package org.deeplearning4j.examples.userInterface

import java.io.File

import org.deeplearning4j.examples.userInterface.util.UIExampleUtils
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.FileStatsStorage

/**
  * A version of UIStorageExample showing how to saved network training data to a file, and then
  * reload it later, to display in in the UI
  *
  * @author Alex Black
  */
object UIStorageExample {

  def main(args: Array[String]) {

    //Run this example twice - once with collectStats = true, and then again with collectStats = false
    val collectStats = true

    val statsFile = new File("UIStorageExampleStats.dl4j")

    if (collectStats) {
      //First run training stats from the network
      //Note that we don't have to actually plot it when we collect it - though we can do that too, if required

      val net = UIExampleUtils.getMnistNetwork
      val trainData = UIExampleUtils.getMnistData

      val statsStorage = new FileStatsStorage(statsFile)
      net.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10))

      net.fit(trainData)

      println("Done")
    } else {
      //Second run: Load the saved stats and visualize. Go to http://localhost:9000/train

      val statsStorage = new FileStatsStorage(statsFile) //If file already exists: load the data from it
      val uiServer = UIServer.getInstance
      uiServer.attach(statsStorage)
    }
  }
}
