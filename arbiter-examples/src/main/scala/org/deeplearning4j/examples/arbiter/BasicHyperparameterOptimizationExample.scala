package org.deeplearning4j.examples.arbiter

import java.io.File
import java.util.concurrent.TimeUnit

import org.deeplearning4j.arbiter.data.DataSetIteratorProvider
import org.deeplearning4j.arbiter.layers.{DenseLayerSpace, OutputLayerSpace}
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver
import org.deeplearning4j.arbiter.optimize.api.termination.{MaxCandidatesCondition, MaxTimeCondition}
import org.deeplearning4j.arbiter.optimize.candidategenerator.RandomSearchGenerator
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner
import org.deeplearning4j.arbiter.optimize.ui.ArbiterUIServer
import org.deeplearning4j.arbiter.optimize.ui.listener.UIOptimizationRunnerStatusListener
import org.deeplearning4j.arbiter.saver.local.multilayer.LocalMultiLayerNetworkSaver
import org.deeplearning4j.arbiter.scoring.multilayer.TestSetAccuracyScoreFunction
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator
import org.deeplearning4j.arbiter.{DL4JConfiguration, MultiLayerSpace}
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * This is a basic hyperparameter optimization example using Arbiter to conduct random search on two network hyperparameters.
  * The two hyperparameters are learning rate and layer size, and the search is conducted for a simple multi-layer perceptron
  * on MNIST data.
  *
  * Note that this example has a UI, but it (currently) does not start automatically.
  * By default, the UI is accessible at http://localhost:8080/arbiter
  *
  * @author Alex Black
  */
object BasicHyperparameterOptimizationExample {

  @throws[Exception]
  def main(args: Array[String]): Unit = {

    //First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
    // fixed values or values to optimize, for each hyperparameter
    val learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1)// Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
    val layerSizeHyperparam = new IntegerParameterSpace(16, 256)           // Integer values will be generated uniformly at random between 16 and 256 (inclusive)

    val hyperparameterSpace: MultiLayerSpace = new MultiLayerSpace.Builder()
       //These next few options: fixed values for all models
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .regularization(true)
      .l2(0.0001)
       //Learning rate: this is something we want to test different values for
      .learningRate(learningRateHyperparam)
      .addLayer( new DenseLayerSpace.Builder()
         //Fixed values for this layer:
        .nIn(784)   //Fixed input: 28x28=784 pixels for MNIST
        .activation("relu")
         //One hyperparameter to infer: layer size
        .nOut(layerSizeHyperparam)
        .build)
      .addLayer( new OutputLayerSpace.Builder()
        //nIn: set the same hyperparemeter as the nOut for the last layer.
        .nIn(layerSizeHyperparam)
         //The remaining hyperparameters: fixed for the output layer
        .nOut(10)
        .activation("softmax")
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .build)
      .pretrain(false).backprop(true).build


    //Now: We need to define a few configuration options
    // (a) How are we going to generate candidates? (random search or grid search)
    val candidateGenerator = new RandomSearchGenerator[DL4JConfiguration](hyperparameterSpace)

    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);
    // (b) How are going to provide data? For now, we'll use a simple built-in data provider for DataSetIterators
    val nTrainEpochs: Int = 2
    val mnistTrain = new MultipleEpochsIterator(nTrainEpochs, new MnistDataSetIterator(64, true, 12345))
    val mnistTest = new MnistDataSetIterator(64, false, 12345)
    val dataProvider = new DataSetIteratorProvider(mnistTrain, mnistTest)

    // (c) How we are going to save the models that are generated and tested?
    //     In this example, let's save them to disk the working directory
    //     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
    val baseSaveDirectory: String = "arbiterExample/"
    val f: File = new File(baseSaveDirectory)
    if (f.exists) f.delete
    f.mkdir
    val modelSaver: ResultSaver[DL4JConfiguration, MultiLayerNetwork, Object] = new LocalMultiLayerNetworkSaver(baseSaveDirectory)

    // (d) What are we actually trying to optimize?
    //     In this example, let's use classification accuracy on the test set
    val scoreFunction = new TestSetAccuracyScoreFunction

    // (e) When should we stop searching? Specify this with termination conditions
    //     For this example, we are stopping the search at 15 minutes or 20 candidates - whichever comes first
    val terminationConditions = Array(new MaxTimeCondition(15, TimeUnit.MINUTES), new MaxCandidatesCondition(20))



    //Given these configuration options, let's put them all together:
    val configuration = new OptimizationConfiguration.Builder[DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object]()
      .candidateGenerator(candidateGenerator)
      .dataProvider(dataProvider)
      .modelSaver(modelSaver)
      .scoreFunction(scoreFunction)
      .terminationConditions(terminationConditions:_*)
      .build

    //And set up execution locally on this machine:
    val runner = new LocalOptimizationRunner[DL4JConfiguration, MultiLayerNetwork, DataSetIterator, Object](configuration, new MultiLayerNetworkTaskCreator[Object])

    //Start the UI
    val server = ArbiterUIServer.getInstance
    runner.addListeners(new UIOptimizationRunnerStatusListener(server))


    //Start the hyperparameter optimization
    runner.execute()


    //Print out some basic stats regarding the optimization procedure
    val sb = new StringBuilder
    sb.append("Best score: ").append(runner.bestScore).append("\n")
      .append("Index of model with best score: ").append(runner.bestScoreCandidateIndex).append("\n")
      .append("Number of configurations evaluated: ").append(runner.numCandidatesCompleted).append("\n")
    println(sb.toString)


    //Get all results, and print out details of the best result:
    val indexOfBestResult: Int = runner.bestScoreCandidateIndex
    val allResults = runner.getResults

    val bestResult = allResults.get(indexOfBestResult).getResult
    val bestModel = bestResult.getResult

    println("\n\nConfiguration of best model:\n")
    println(bestModel.getLayerWiseConfigurations.toJson)


    //Note: UI server will shut down once execution is complete, as JVM will exit
    //So do a Thread.sleep(1 minute) to keep JVM alive, so that network configurations can be viewed
    Thread.sleep(60000)
    System.exit(0)
  }
}
