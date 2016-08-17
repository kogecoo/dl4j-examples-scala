package org.deeplearning4j.examples.feedforward.anomalydetection

import java.awt._
import java.awt.image.BufferedImage
import java.util._
import javax.swing._

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable


object MNISTAnomalyExample {

  def main(args: Array[String]) = {

    //Set up network. 784 in/out (as MNIST images are 28x28).
    //784 -> 250 -> 10 -> 250 -> 784
    val conf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .iterations(1)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.ADAGRAD)
      .activation("relu")
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.05)
      .regularization(true).l2(0.001)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
        .build())
      .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
        .lossFunction(LossFunctions.LossFunction.MSE)
        .build())
      .pretrain(false).backprop(true)
      .build()

    val net = new MultiLayerNetwork(conf)
    net.setListeners(Collections.singletonList[IterationListener](new ScoreIterationListener(1)))

    //Load data and split into training and testing sets. 40000 train, 10000 test
    val iter = new MnistDataSetIterator(100,50000,false)

    val featuresTrainBuilder = mutable.ArrayBuilder.make[INDArray]
    val featuresTestBuilder  = mutable.ArrayBuilder.make[INDArray]
    val labelsTestBuilder    = mutable.ArrayBuilder.make[INDArray]

    val r = new Random(12345)
    while(iter.hasNext){
      val ds = iter.next()
      val split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
      featuresTrainBuilder += split.getTrain.getFeatureMatrix
      val dsTest = split.getTest
      featuresTestBuilder += dsTest.getFeatureMatrix
      val indexes = Nd4j.argMax(dsTest.getLabels, 1); //Convert from one-hot representation -> index
      labelsTestBuilder += indexes
    }

    val featuresTrain= featuresTrainBuilder.result()
    val featuresTest = featuresTestBuilder.result()
    val labelsTest   = labelsTestBuilder.result()

    //Train model:
    val nEpochs = 30
    (0 until nEpochs).foreach { epoch =>
      featuresTrain.foreach { data =>
        net.fit(data,data)
      }
      System.out.println("Epoch " + epoch + " complete")
    }

    //Evaluate the model on test data
    //Score each digit/example in test set separately
    //Then add triple (score, digit, and INDArray data) to lists and sort by score
    //This allows us to get best N and worst N digits for each type
    val listsByDigit = mutable.Map.empty[Int, mutable.ArrayBuffer[(Double, Int, INDArray)]]
    (0 until 10).foreach { i =>
      listsByDigit.update(i, mutable.ArrayBuffer.empty[(Double, Int, INDArray)])
    }

    var count = 0
    featuresTest.indices.foreach { i =>
      val testData = featuresTest(i)
      val labels   = labelsTest(i)
      val nRows    = testData.rows()
      (0 until nRows).foreach { j =>
        val example: INDArray = testData.getRow(j)
        val label = labels.getDouble(j).toInt
        val score = net.score(new DataSet(example, example))
        val arr = listsByDigit.getOrElseUpdate(label, mutable.ArrayBuffer.empty[(Double, Int, INDArray)])
        arr += ((score, count, example))
        count += 1
      }
    }

    //Sort data by score, separately for each digit
    val sorted = listsByDigit.map({ case (k, vs) => (k, vs.sortBy(_._1).toSeq) }).toMap

    //Select the 5 best and 5 worst numbers (by reconstruction error) for each digit
    val bestBuilder = mutable.ArrayBuilder.make[INDArray]
    val worstBuilder = mutable.ArrayBuilder.make[INDArray]
    (0 until 10).foreach { i =>
      val list = sorted.get(i).get
      (0 until 5).foreach { j =>
        bestBuilder += list(j)._3
        worstBuilder += list(list.size - j - 1)._3
      }
    }
    val best = bestBuilder.result()
    val worst = worstBuilder.result()

    //Visualize the best and worst digits
    val bestVisualizer = new MNISTVisualizer(2.0,best,"Best (Low Rec. Error)")
    bestVisualizer.visualize()

    val worstVisualizer = new MNISTVisualizer(2.0,worst,"Worst (High Rec. Error)")
    worstVisualizer.visualize()
  }

  private[this] class MNISTVisualizer(
    imageScale: Double,
    digits: Seq[INDArray], //Digits (as row vectors), one per INDArray
    title: String
  ) {

    def visualize(): Unit = {
      val frame = new JFrame()
      frame.setTitle(title)
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)

      val panel = new JPanel()
      panel.setLayout(new GridLayout(0,5))

      val images: Seq[JLabel] = getComponents
      images.foreach { image =>
        panel.add(image)
      }

      frame.add(panel)
      frame.setVisible(true)
      frame.pack()
    }

    private[this] def getComponents: Seq[JLabel] = {
      digits.map { arr =>
        val bi = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
        (0 until 768).foreach { i =>
          bi.getRaster.setSample(i % 28, i / 28, 0, 255 * arr.getDouble(i).toInt)
        }
        val orig = new ImageIcon(bi)
        val imageScaled: Image = orig.getImage.getScaledInstance((imageScale * 28).asInstanceOf[Int], (imageScale * 28).asInstanceOf[Int], Image.SCALE_REPLICATE)
        val scaled = new ImageIcon(imageScaled)
        new JLabel(scaled)
      }
    }
  }
}
