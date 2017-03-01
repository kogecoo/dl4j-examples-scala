package org.deeplearning4j.examples.feedforward.anomalydetection

import java.awt.{GridLayout, Image}

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.awt.image.BufferedImage
import java.util.{Collections, Random}
import javax.swing._

import scala.collection.mutable

/** Example: Anomaly Detection on MNIST using simple autoencoder without pretraining
  * The goal is to identify outliers digits, i.e., those digits that are unusual or
  * not like the typical digits.
  * This is accomplished in this example by using reconstruction error: stereotypical
  * examples should have low reconstruction error, whereas outliers should have high
  * reconstruction error
  *
  * @author Alex Black
  */
object MNISTAnomalyExample {
  @throws[Exception]
  def main(args: Array[String]) {
    //Set up network. 784 in/out (as MNIST images are 28x28).
    //784 -> 250 -> 10 -> 250 -> 784
    val conf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.ADAGRAD)
      .activation(Activation.RELU)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.05)
      .regularization(true)
      .l2(0.0001)
      .list
      .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
        .build)
      .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
        .build)
      .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
        .build)
      .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
        .lossFunction(LossFunctions.LossFunction.MSE)
        .build)
      .pretrain(false).backprop(true)
      .build

    val net = new MultiLayerNetwork(conf)
    net.setListeners(Collections.singletonList(new ScoreIterationListener(1).asInstanceOf[IterationListener]))

    //Load data and split into training and testing sets. 40000 train, 10000 test
    val iter = new MnistDataSetIterator(100, 50000, false)

    val featuresTrain = mutable.ArrayBuffer.empty[INDArray]
    val featuresTest = mutable.ArrayBuffer.empty[INDArray]
    val labelsTest = mutable.ArrayBuffer.empty[INDArray]

    val r = new Random(12345)
    while (iter.hasNext) {
      val ds = iter.next
      val split = ds.splitTestAndTrain(80, r) //80/20 split (from miniBatch = 100)
      featuresTrain += split.getTrain.getFeatureMatrix
      val dsTest = split.getTest
      featuresTest += dsTest.getFeatureMatrix
      val indexes = Nd4j.argMax(dsTest.getLabels, 1) //Convert from one-hot representation -> index
      labelsTest += indexes
    }

    //Train model:
    val nEpochs = 30
    for (epoch <- 0 until nEpochs) {
      for (data <- featuresTrain) {
        net.fit(data, data)
      }
      println("Epoch " + epoch + " complete")
    }
    //Evaluate the model on test data
    //Score each digit/example in test set separately
    //Then add triple (score, digit, and INDArray data) to lists and sort by score
    //This allows us to get best N and worst N digits for each type
    val listsByDigit = mutable.Map.empty[Integer, mutable.ArrayBuffer[(Double, Integer, INDArray)]]
    for (i <- 0 until 10) {
      listsByDigit.put(i, mutable.ArrayBuffer.empty[(Double, Integer, INDArray)])
    }

    var count = 0
    for (i <- featuresTest.indices) {
      val testData = featuresTest(i)
      val labels = labelsTest(i)
      val nRows = testData.rows
      for (j <- 0 until nRows) {
        val example = testData.getRow(j)
        val label = labels.getDouble(j).toInt
        val score = net.score(new DataSet(example, example))
        listsByDigit(label) += ((score, count, example))
        count += 1
      }
    }
    //Sort data by score, separately for each digit
    val sortedListsByDigit = listsByDigit.map({ case (k, list) =>
      (k, list.sortBy(_._1).toList)
    }).toMap

    //Select the 5 best and 5 worst numbers (by reconstruction error) for each digit
    val best = mutable.ArrayBuffer.empty[INDArray]
    val worst = mutable.ArrayBuffer.empty[INDArray]
    for (i <- 0 until 10) {
      for (j <- 0 until 5) {
        val list = sortedListsByDigit(i)
        best += list(j)._3
        worst += list(list.size - j - 1)._3
      }
    }

    //Visualize the best and worst digits
    val bestVisualizer = new MNISTAnomalyExample.MNISTVisualizer(2.0, best.toArray, "Best (Low Rec. Error)")
    bestVisualizer.visualize()
    val worstVisualizer = new MNISTAnomalyExample.MNISTVisualizer(2.0, worst.toArray, "Worst (High Rec. Error)")
    worstVisualizer.visualize()
  }

  class MNISTVisualizer(imageScale: Double,
                        digits: Array[INDArray], //Digits (as row vectors), one per INDArray
                        title: String,
                        gridWidth: Int = 5) {

    def visualize() {
      val frame: JFrame = new JFrame
      frame.setTitle(title)
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      val panel: JPanel = new JPanel
      panel.setLayout(new GridLayout(0, gridWidth))
      val list  = getComponents
      for (image <- list) {
        panel.add(image)
      }
      frame.add(panel)
      frame.setVisible(true)
      frame.pack()
    }

    private def getComponents: List[JLabel] = {
      val images = mutable.ArrayBuffer.empty[JLabel]
      for (arr <- digits) {
        val bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
        for (i <- 0 until 784) {
          bi.getRaster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i)).toInt)
        }
        val orig = new ImageIcon(bi)
        val imageScaled = orig.getImage.getScaledInstance((imageScale * 28).toInt, (imageScale * 28).toInt, Image.SCALE_REPLICATE)
        val scaled = new ImageIcon(imageScaled)
        images += new JLabel(scaled)
      }
      images.toList
    }
  }

}
