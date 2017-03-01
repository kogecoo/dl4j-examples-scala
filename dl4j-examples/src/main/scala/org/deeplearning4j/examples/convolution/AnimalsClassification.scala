package org.deeplearning4j.examples.convolution

import org.apache.commons.io.FilenameUtils
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.{FlipImageTransform, ImageTransform, WarpImageTransform}
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.distribution.{Distribution, GaussianDistribution, NormalDistribution}
import org.deeplearning4j.nn.conf.inputs.{InputType, InvalidInputTypeException}
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.NetSaverLoaderUtils
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.{DataNormalization, ImagePreProcessingScaler}
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import java.io.File
import java.util.Random

import scala.collection.JavaConverters._

/**
  * Animal Classification
  *
  * Example classification of photos from 4 different animals (bear, duck, deer, turtle).
  *
  * References:
  *  - U.S. Fish and Wildlife Service (animal sample dataset): http://digitalmedia.fws.gov/cdm/
  *  - Tiny ImageNet Classification with CNN: http://cs231n.stanford.edu/reports/leonyao_final.pdf
  *
  * CHALLENGE: Current setup gets low score results. Can you improve the scores? Some approaches:
  *  - Add additional images to the dataset
  *  - Apply more transforms to dataset
  *  - Increase epochs
  *  - Try different model configurations
  *  - Tune by adjusting learning rate, updaters, activation & loss functions, regularization, ...
  */
object AnimalsClassification {
  protected val log: Logger = LoggerFactory.getLogger(AnimalsClassification.getClass)
  protected var height: Int = 100
  protected var width: Int = 100
  protected var channels: Int = 3
  protected var numExamples: Int = 80
  protected var numLabels: Int = 4
  protected var batchSize: Int = 20
  protected var seed: Long = 42
  protected var rng: Random = new Random(seed)
  protected var listenerFreq: Int = 1
  protected var iterations: Int = 1
  protected var epochs: Int = 50
  protected var splitTrainTest: Double = 0.8
  protected var nCores: Int = 2
  protected var save: Boolean = false
  protected var modelType: String = "AlexNet" // LeNet, AlexNet or Custom but you need to fill it out

  def customModel: MultiLayerNetwork = {
    /**
      * Use this method to build your own custom model.
      **/
    null
  }

  @throws[Exception]
  def main(args: Array[String]) {
    new AnimalsClassification().run(args)
  }

}

class AnimalsClassification {

  import AnimalsClassification._

  @throws[Exception]
  def run(args: Array[String]) {

    log.info("Load data....")
    /** cd
      * Data Setup -> organize and limit data file paths:
      *  - mainPath = path to image files
      *  - fileSplit = define basic dataset split with limits on format
      *  - pathFilter = define additional file load filter to limit size and balance batch content
      * */
    val labelMaker = new ParentPathLabelGenerator
    val mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals/")
    val fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng)
    val pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize)

    /**
      * Data Setup -> train test split
      *  - inputSplit = define train and test split
      **/
    val inputSplit = fileSplit.sample(pathFilter, numExamples * (1 + splitTrainTest), numExamples * (1 - splitTrainTest))
    val trainData = inputSplit(0)
    val testData = inputSplit(1)

    /**
      * Data Setup -> transformation
      *  - Transform = how to tranform images and generate large dataset to train on
      **/
    val flipTransform1: ImageTransform = new FlipImageTransform(rng)
    val flipTransform2: ImageTransform = new FlipImageTransform(new Random(123))
    val warpTransform: ImageTransform = new WarpImageTransform(rng, 42)
    //        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
    val transforms = List[ImageTransform](flipTransform1, warpTransform, flipTransform2).asJava

    /**
      * Data Setup -> normalization
      *  - how to normalize images and generate large dataset to train on
      **/
    val scaler: DataNormalization = new ImagePreProcessingScaler(0, 1)

    log.info("Build model....")

    // Uncomment below to try AlexNet. Note change height and width to at least 100
    //        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();

    val network = modelType match {
      case "LeNet"   => lenetModel
      case "AlexNet" => alexnetModel
      case "custom"  => customModel
      case _         => throw new InvalidInputTypeException("Incorrect model provided.")
    }
    network.init()
    network.setListeners(new ScoreIterationListener(listenerFreq))

    /**
      * Data Setup -> define how to load data into net:
      *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
      *  - dataIter = a generator that only loads one batch at a time into memory to save memory
      *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
      **/
    val recordReader: ImageRecordReader = new ImageRecordReader(height, width, channels, labelMaker)
    var dataIter: DataSetIterator = null
    var trainIter: MultipleEpochsIterator = null


    log.info("Train model....")
    // Train without transformations
    recordReader.initialize(trainData, null)
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores)
    network.fit(trainIter)

    // Train with transformations
    import scala.collection.JavaConversions._
    for (transform <- transforms) {
      print("\nTraining on transformation: " + transform.getClass.toString + "\n\n")
      recordReader.initialize(trainData, transform)
      dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
      scaler.fit(dataIter)
      dataIter.setPreProcessor(scaler)
      trainIter = new MultipleEpochsIterator(epochs, dataIter, nCores)
      network.fit(trainIter)
    }

    log.info("Evaluate model....")
    recordReader.initialize(testData)
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels)
    scaler.fit(dataIter)
    dataIter.setPreProcessor(scaler)
    val eval: Evaluation = network.evaluate(dataIter)
    log.info(eval.stats(true))

    // Example on how to get predict results with trained model
    dataIter.reset()
    val testDataSet: DataSet = dataIter.next
    val expectedResult = testDataSet.getLabelName(0)
    val predict = network.predict(testDataSet)
    val modelResult = predict.get(0)
    print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n")
    if (save) {
      log.info("Save model....")
      val basePath = FilenameUtils.concat(System.getProperty("user.dir"), "src/main/resources/")
      NetSaverLoaderUtils.saveNetworkAndParameters(network, basePath)
      NetSaverLoaderUtils.saveUpdators(network, basePath)
    }
    log.info("****************Example finished********************")
  }

  private def convInit(name: String, in: Int, out: Int, kernel: Array[Int], stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build()
  }

  private def conv3x3(name: String, out: Int, bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(Array[Int](3, 3), Array[Int](1, 1), Array[Int](1, 1)).name(name).nOut(out).biasInit(bias).build()
  }

  private def conv5x5(name: String, out: Int, stride: Array[Int], pad: Array[Int], bias: Double): ConvolutionLayer = {
    new ConvolutionLayer.Builder(Array[Int](5, 5), stride, pad).name(name).nOut(out).biasInit(bias).build()
  }

  private def maxPool(name: String, kernel: Array[Int]): SubsamplingLayer = {
    new SubsamplingLayer.Builder(kernel, Array[Int](2, 2)).name(name).build()
  }

  private def fullyConnected(name: String, out: Int, bias: Double, dropOut: Double, dist: Distribution): DenseLayer = {
    new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build()
  }

  def lenetModel: MultiLayerNetwork = {
    /**
      * Revisde Lenet Model approach developed by ramgo2 achieves slightly above random
      * Reference: https://gist.github.com/ramgo2/833f12e92359a2da9e5c2fb6333351c5
      **/
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(false)
      .l2(0.005)
      .activation(Activation.RELU) // tried 0.0001, 0.0005
      .learningRate(0.0001)
      .weightInit(WeightInit.XAVIER) // tried 0.00001, 0.00005, 0.000001
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.RMSPROP)
      .momentum(0.9)
      .list()
      .layer(0, convInit("cnn1", channels, 50, Array[Int](5, 5), Array[Int](1, 1), Array[Int](0, 0), 0))
      .layer(1, maxPool("maxpool1", Array[Int](2, 2)))
      .layer(2, conv5x5("cnn2", 100, Array[Int](5, 5), Array[Int](1, 1), 0))
      .layer(3, maxPool("maxool2", Array[Int](2, 2)))
      .layer(4, new DenseLayer.Builder().nOut(500).build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(numLabels)
        .activation(Activation.SOFTMAX)
        .build())
      .backprop(true).pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build()
    new MultiLayerNetwork(conf)
  }

  def alexnetModel: MultiLayerNetwork = {
    /**
      * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
      * and the imagenetExample code referenced.
      * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
      **/
    val nonZeroBias: Double = 1
    val dropOut: Double = 0.5
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .weightInit(WeightInit.DISTRIBUTION)
      .dist(new NormalDistribution(0.0, 0.01))
      .activation(Activation.RELU)
      .updater(Updater.NESTEROVS)
      .iterations(iterations)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // normalize to prevent vanishing or exploding gradients
      .learningRate(1e-2)
      .biasLearningRate(1e-2 * 2)
      .learningRateDecayPolicy(LearningRatePolicy.Step)
      .lrPolicyDecayRate(0.1)
      .lrPolicySteps(100000)
      .regularization(true)
      .l2(5 * 1e-4)
      .momentum(0.9)
      .miniBatch(false)
      .list()
      .layer(0, convInit("cnn1", channels, 96, Array[Int](11, 11), Array[Int](4, 4), Array[Int](3, 3), 0))
      .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build)
      .layer(2, maxPool("maxpool1", Array[Int](3, 3)))
      .layer(3, conv5x5("cnn2", 256, Array[Int](1, 1), Array[Int](2, 2), nonZeroBias))
      .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build)
      .layer(5, maxPool("maxpool2", Array[Int](3, 3)))
      .layer(6, conv3x3("cnn3", 384, 0))
      .layer(7, conv3x3("cnn4", 384, nonZeroBias))
      .layer(8, conv3x3("cnn5", 256, nonZeroBias))
      .layer(9, maxPool("maxpool3", Array[Int](3, 3)))
      .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
      .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .name("output")
        .nOut(numLabels)
        .activation(Activation.SOFTMAX)
        .build())
      .backprop(true).pretrain(false)
      .setInputType(InputType.convolutional(height, width, channels))
      .build()
    new MultiLayerNetwork(conf)
  }
}
