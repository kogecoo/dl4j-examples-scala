/**
  * This program trains a RNN to predict category of a news headlines. It uses word vector generated from PrepareWordVector.java.
  * - Labeled News are stored in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder in train and test folders.
  * - categories.txt file in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder contains category code and description.
  * - This categories are used along with actual news for training.
  * - news word vector is contained  in \dl4j-examples\src\main\resources\NewsData\NewsWordVector.txt file.
  * - Trained model is stored in \dl4j-examples\src\main\resources\NewsData\NewsModel.net file
  * - News Data contains only 3 categories currently.
  * - Data set structure is as given below
  * - categories.txt - this file contains various categories in category id,category description format. Sample categories are as below
  * 0,crime
  * 1,politics
  * 2,bollywood
  * 3,Business&Development
  * - For each category id above, there is a file containig actual news headlines, e.g.
  * 0.txt - contains news for crime headlines
  * 1.txt - contains news for politics headlines
  * 2.txt - contains news for bollywood
  * 3.txt - contains news for Business&Development
  * - You can add any new category by adding one line in categories.txt and respective news file in folder mentioned above.
  * - Below are training results with the news data given with this example.
  * ==========================Scores========================================
  * Accuracy:        0.9343
  * Precision:       0.9249
  * Recall:          0.9327
  * F1 Score:        0.9288
  * ========================================================================
  * <p>
  * Note :
  * - This code is a modification of original example named Word2VecSentimentRNN.java
  * - Results may vary with the data you use to train this network
  * <p>
  * <b>KIT Solutions Pvt. Ltd. (www.kitsol.com)</b>
  **//**
  * This program trains a RNN to predict category of a news headlines. It uses word vector generated from PrepareWordVector.java.
  * - Labeled News are stored in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder in train and test folders.
  * - categories.txt file in \dl4j-examples\src\main\resources\NewsData\LabelledNews folder contains category code and description.
  * - This categories are used along with actual news for training.
  * - news word vector is contained  in \dl4j-examples\src\main\resources\NewsData\NewsWordVector.txt file.
  * - Trained model is stored in \dl4j-examples\src\main\resources\NewsData\NewsModel.net file
  * - News Data contains only 3 categories currently.
  * - Data set structure is as given below
  * - categories.txt - this file contains various categories in category id,category description format. Sample categories are as below
  * 0,crime
  * 1,politics
  * 2,bollywood
  * 3,Business&Development
  * - For each category id above, there is a file containig actual news headlines, e.g.
  * 0.txt - contains news for crime headlines
  * 1.txt - contains news for politics headlines
  * 2.txt - contains news for bollywood
  * 3.txt - contains news for Business&Development
  * - You can add any new category by adding one line in categories.txt and respective news file in folder mentioned above.
  * - Below are training results with the news data given with this example.
  * ==========================Scores========================================
  * Accuracy:        0.9343
  * Precision:       0.9249
  * Recall:          0.9327
  * F1 Score:        0.9288
  * ========================================================================
  * <p>
  * Note :
  * - This code is a modification of original example named Word2VecSentimentRNN.java
  * - Results may vary with the data you use to train this network
  * <p>
  * <b>KIT Solutions Pvt. Ltd. (www.kitsol.com)</b>
  */
package org.deeplearning4j.examples.recurrent.processnews

import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.{DefaultTokenizerFactory, TokenizerFactory}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

object TrainNews {

  @throws[Exception]
  def main(args: Array[String]) {
    val userDirectory = new ClassPathResource("NewsData").getFile.getAbsolutePath + File.separator
    val DATA_PATH = userDirectory + "LabelledNews"
    val WORD_VECTORS_PATH = userDirectory + "NewsWordVector.txt"

    val batchSize: Int = 50 //Number of examples in each minibatch
    val nEpochs: Int = 1000 //Number of epochs (full passes of training data) to train on
    val truncateReviewsToLength: Int = 300 //Truncate reviews with length (# words) greater than this

    //DataSetIterators for training and testing respectively
    //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
    val wordVectors = WordVectorSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH))

    var tokenizerFactory: TokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

    val iTrain = new NewsIterator.Builder()
      .dataDirectory(DATA_PATH)
      .wordVectors(wordVectors)
      .batchSize(batchSize)
      .truncateLength(truncateReviewsToLength)
      .tokenizerFactory(tokenizerFactory)
      .train(true)
      .build

    println(s"${iTrain.dataDirectory}," +
      s" ${iTrain.wordVectors}," +
      s" ${iTrain.batchSize}," +
      s" ${iTrain.truncateLength}," +
      s" ${iTrain.train}," +
      s"${iTrain.tokenizerFactory}")

    val iTest = new NewsIterator.Builder()
      .dataDirectory(DATA_PATH)
      .wordVectors(wordVectors)
      .batchSize(batchSize)
      .tokenizerFactory(tokenizerFactory)
      .truncateLength(truncateReviewsToLength)
      .train(false)
      .build
    println(s"${iTest.dataDirectory}," +
      s" ${iTest.wordVectors}," +
      s" ${iTest.batchSize}," +
      s" ${iTest.truncateLength}," +
      s" ${iTest.train}," +
      s"${iTest.tokenizerFactory}")

    //DataSetIterator train = new AsyncDataSetIterator(iTrain,1);
    //DataSetIterator test = new AsyncDataSetIterator(iTest,1);

    val inputNeurons: Int = wordVectors.getWordVector(wordVectors.vocab.wordAtIndex(0)).length // 100 in our case
    val outputs: Int = iTrain.getLabels.size

    tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)
    //Set up network configuration
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .updater(Updater.RMSPROP)
      .regularization(true).l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
      .gradientNormalizationThreshold(1.0)
      .learningRate(0.0018)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(inputNeurons).nOut(200)
        .activation("softsign").build())
      .layer(1, new RnnOutputLayer.Builder().activation("softmax")
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .nIn(200).nOut(outputs).build())
      .pretrain(false).backprop(true).build()

    val net: MultiLayerNetwork = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))

    println("Starting training")
    for (i <- 0 until nEpochs) {
      net.fit(iTrain)
      iTrain.reset()
      println("Epoch " + i + " complete. Starting evaluation:")

      //Run evaluation. This is on 25k reviews, so can take some time
      val evaluation: Evaluation = new Evaluation
      while (iTest.hasNext) {
        val t = iTest.next
        val features = t.getFeatureMatrix
        val lables = t.getLabels
        //println("labels : " + lables);
        val inMask = t.getFeaturesMaskArray
        val outMask = t.getLabelsMaskArray
        val predicted = net.output(features, false)

        //println("predicted : " + predicted);
        evaluation.evalTimeSeries(lables, predicted, outMask)
      }
      iTest.reset()
      println(evaluation.stats)
    }

    ModelSerializer.writeModel(net, userDirectory + "NewsModel.net", true)
    println("----- Example complete -----")
  }
}
