package org.deeplearning4j.rnn

import java.io.{File, IOException}
import java.net.URL
import java.nio.charset.Charset
import java.nio.file.Files
import java.util.{Collections, Random}

import com.beust.jcommander.{JCommander, Parameter, ParameterException}
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.broadcast.Broadcast
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{BackpropType, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable

/**
  * GravesLSTM + Spark character modelling example
  * Example: Train a LSTM RNN to generates text, one character at a time.
  * Training here is done on Spark
  *
  * See dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/GravesLSTMCharModellingExample.java
  * for the single-machine version of this example
  *
  * To run the example locally: Run the example as-is. The example is set up to use Spark local by default.
  * NOTE: Spark local should only be used for development/testing. For data parallel training on a single machine
  * (for example, multi-GPU systems) instead use ParallelWrapper (which is faster than using Spark for training on a single machine).
  * See for example MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
  *
  * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
  * OR first modify the example by setting the field "useSparkLocal = false"
  *
  * @author Alex Black
  */
object SparkLSTMCharacterExample {

  private val INT_TO_CHAR: Map[Int, Char] = SparkLSTMCharacterExample.getIntToChar
  private val CHAR_TO_INT: Map[Char, Int] = SparkLSTMCharacterExample.getCharToInt
  private val N_CHARS = INT_TO_CHAR.size
  private val nOut = CHAR_TO_INT.size
  private val exampleLength = 1000 //Length of each training example sequence to use

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    new SparkLSTMCharacterExample().entryPoint(args)
  }

  /**
    * Load data from a file, and remove any invalid characters.
    * Data is returned as a single large String
    */
  @throws[IOException]
  private def getDataAsString(filePath: String): String = {
    val lines = Files.readAllLines(new File(filePath).toPath, Charset.defaultCharset)
    val sb = new StringBuilder
    import scala.collection.JavaConversions._
    for (line <- lines) {
      val chars: Array[Char] = line.toCharArray
      chars.indices.foreach { i =>
        if (SparkLSTMCharacterExample.CHAR_TO_INT.containsKey(chars(i))) sb.append(chars(i))
      }
      sb.append("\n")
    }
    sb.toString
  }

  /**
    * Get the training data - a JavaRDD<DataSet>
    * Note that this approach for getting training data is a special case for this example (modelling characters), and
    * should  not be taken as best practice for loading data (like CSV etc) in general.
    */
  @throws[IOException]
  def getTrainingData(sc: JavaSparkContext): JavaRDD[DataSet] = {
    //Get data. For the sake of this example, we are doing the following operations:
    // File -> String -> List<String> (split into length "sequenceLength" characters) -> JavaRDD<String> -> JavaRDD<DataSet>
    val list = getShakespeareAsList(exampleLength)
    val rawStrings = sc.parallelize(list)
    val bcCharToInt = sc.broadcast(CHAR_TO_INT)
    rawStrings.map(stringToDataSetFn(bcCharToInt))
  }

  @throws[Exception]
  def stringToDataSetFn(ctiBroadcast: Broadcast[Map[Char, Int]])(s: String): DataSet = {
    //Here: take a String, and map the characters to a one-hot representation
    val cti: Map[Char, Int] = ctiBroadcast.value
    val length: Int = s.length
    val features: INDArray = Nd4j.zeros(1, N_CHARS, length - 1)
    val labels: INDArray = Nd4j.zeros(1, N_CHARS, length - 1)
    val chars: Array[Char] = s.toCharArray
    val f: Array[Int] = new Array[Int](3)
    val l: Array[Int] = new Array[Int](3)
    var i: Int = 0
    (0 until chars.length - 2).foreach { i =>
      f(1) = cti(chars(i))
      f(2) = i
      l(1) = cti(chars(i + 1)) //Predict the next character given past and current characters
      l(2) = i
      features.putScalar(f, 1.0)
      labels.putScalar(l, 1.0)
    }
    new DataSet(features, labels)
  }
  //This function downloads (if necessary), loads and splits the raw text data into "sequenceLength" strings
  @throws[IOException]
  private def getShakespeareAsList(sequenceLength: Int): List[String] = {
    //The Complete Works of William Shakespeare
    //5.3MB file in UTF-8 Encoding, ~5.4 million characters
    //https://www.gutenberg.org/ebooks/100
    val url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt"
    val tempDir = System.getProperty("java.io.tmpdir")
    val fileLocation = tempDir + "/Shakespeare.txt" //Storage location from downloaded file
    val f = new File(fileLocation)
    if (!f.exists) {
      FileUtils.copyURLToFile(new URL(url), f)
      println("File downloaded to " + f.getAbsolutePath)
    } else {
      println("Using existing text file at " + f.getAbsolutePath)
    }

    if (!f.exists) throw new IOException("File does not exist: " + fileLocation) //Download problem?

    val allData = getDataAsString(fileLocation)

    val list = mutable.ArrayBuffer.empty[String]
    val length = allData.length
    var currIdx = 0
    while (currIdx + sequenceLength < length) {
      val end = currIdx + sequenceLength
      val substr: String = allData.substring(currIdx, end)
      currIdx = end
      list += substr
    }
    list.toList
  }

  /**
    * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
    */
  private def getValidCharacters: Array[Char] = {
    val validChars = mutable.ArrayBuffer.empty[Char]
    val chars = ('a' to 'z') ++
      ('A' to 'Z') ++
      ('0' to '9') ++
      Seq('!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t')

    chars.foreach { c =>
      validChars += c
    }
    validChars.toArray
  }

  def getIntToChar: Map[Int, Char] = {
    val m = mutable.Map.empty[Int, Char]
    val chars = getValidCharacters
    chars.indices.foreach { i =>
      m.update(i, chars(i))
    }
    m.toMap
  }

  def getCharToInt: Map[Char, Int] = {
    val m = mutable.Map.empty[Char, Int]
    val chars = getValidCharacters
    chars.indices.foreach { i =>
      m.update(chars(i), i)
    }
    m.toMap
  }

}

class SparkLSTMCharacterExample {

  private val log: Logger = LoggerFactory.getLogger(classOf[SparkLSTMCharacterExample])

  @Parameter(names = Array("-useSparkLocal"), description = "Use spark local (helper for testing/running without spark submit)", arity = 1) private val useSparkLocal: Boolean = true
  @Parameter(names = Array("-batchSizePerWorker"), description = "Number of examples to fit each worker with") private val batchSizePerWorker: Int = 8 //How many examples should be used per worker (executor) when fitting?
  @Parameter(names = Array("-numEpochs"), description = "Number of epochs for training") private val numEpochs: Int = 1

  @throws[Exception]
  protected def entryPoint(args: Array[String]) {
    //Handle command line arguments
    val jcmdr: JCommander = new JCommander(this, Array():_*)
    try {
      jcmdr.parse(args:_*)
    } catch { case e: ParameterException =>
      //User provides invalid input -> print the usage info
      jcmdr.usage()
      try {
        Thread.sleep(500)
      } catch { case e2: Exception => () }
      throw e
    }

    val rng = new Random(12345)
    val lstmLayerSize: Int = 200 //Number of units in each GravesLSTM layer
    val tbpttLength: Int = 50 //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
    val nSamplesToGenerate: Int = 4 //Number of samples to generate after each training epoch
    val nCharactersToSample: Int = 300 //Length of each sample to generate
    val generationInitialization: String = null //Optional character initialization; a random character is used if null
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default

    //Set up network configuration:
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)
      .list
      .layer(0, new GravesLSTM.Builder().nIn(SparkLSTMCharacterExample.CHAR_TO_INT.size).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
        .nIn(lstmLayerSize).nOut(SparkLSTMCharacterExample.nOut).build) //MCXENT + softmax for classification
      .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
      .pretrain(false).backprop(true)
      .build


    //-------------------------------------------------------------
    //Set up the Spark-specific configuration
    /* How frequently should we average parameters (in number of minibatches)?
    Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
    learning difficulties (i.e., network may not converge) */
    val averagingFrequency: Int = 3

    //Set up Spark configuration and context
    val sparkConf = new SparkConf

    if (useSparkLocal) {
      sparkConf.setMaster("local[*]")
    }
    sparkConf.setAppName("LSTM Character Example")
    val sc = new JavaSparkContext(sparkConf)

    val trainingData = SparkLSTMCharacterExample.getTrainingData(sc)


    //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
    //Here, we are using standard parameter averaging
    //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring
    val examplesPerDataSetObject = 1
    val tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
      .workerPrefetchNumBatches(2)
      .averagingFrequency(averagingFrequency)  //Asynchronously prefetch up to 2 batches
      .batchSizePerWorker(batchSizePerWorker)
      .build
    val sparkNetwork: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, conf, tm)
    sparkNetwork.setListeners(Collections.singletonList[IterationListener](new ScoreIterationListener(1)))

    //Do training, and then generate and print samples from network
    (0 until numEpochs).foreach { i =>
      //Perform one epoch of training. At the end of each epoch, we are returned a copy of the trained network
      val net = sparkNetwork.fit(trainingData)

      //Sample some characters from the network (done locally)
      log.info("Sampling characters from network given initialization \"" +
        (if (generationInitialization == null) "" else generationInitialization) + "\"")
      val samples = sampleCharactersFromNetwork(generationInitialization, net, rng, SparkLSTMCharacterExample.INT_TO_CHAR,
        nCharactersToSample, nSamplesToGenerate)

      samples.indices.foreach { j =>
        log.info("----- Sample " + j + " -----")
        log.info(samples(j))
      }
    }
    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)
    log.info("\n\nExample complete")
  }

  /**
    * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
    * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
    * Note that the initalization is used for all samples
    *
    * @param initialization     String, may be null. If null, select a random character as initialization for all samples
    * @param charactersToSample Number of characters to sample from network (excluding initialization)
    * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
    */
  private def sampleCharactersFromNetwork(initialization: String, net: MultiLayerNetwork, rng: Random, intToChar: Map[Int, Char], charactersToSample: Int, numSamples: Int): Array[String] = {
    //Set up initialization. If no initialization: use a random character
    val _initialization = if (initialization == null) {
      val randomCharIdx = rng.nextInt(intToChar.size)
      String.valueOf(intToChar.get(randomCharIdx))
    } else initialization

    //Create input for initialization
    val initializationInput = Nd4j.zeros(numSamples, intToChar.size, _initialization.length)
    val init = _initialization.toCharArray
    for (i <- init.indices) {
      val idx = SparkLSTMCharacterExample.CHAR_TO_INT(init(i))
      for (j <- 0 until numSamples) {
        initializationInput.putScalar(Array(j, idx, i), 1.0f)
      }
    }
    val sb = new Array[StringBuilder](numSamples)
    for (i <- 0 until numSamples) {
      sb(i) = new StringBuilder(initialization)
    }
    //Sample from network (and feed samples back into input) one character at a time (for all samples)
    //Sampling is done in parallel here
    net.rnnClearPreviousState()
    var output = net.rnnTimeStep(initializationInput)
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0) //Gets the last time step output

    for (i <- 0 until charactersToSample) {
      //Set up next input (single time step) by sampling from previous output
      val nextInput: INDArray = Nd4j.zeros(numSamples, intToChar.size)
      //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
      for (s <- 0 until numSamples) {
        val outputProbDistribution: Array[Double] = new Array[Double](intToChar.size)
          for (j <- outputProbDistribution.indices) {
            outputProbDistribution(j) = output.getDouble(s, j)
          }
          val sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng)
          nextInput.putScalar(Array(s, sampledCharacterIdx), 1.0f) //Prepare next time step input
          sb(s).append(intToChar.get(sampledCharacterIdx)) //Add sampled character to StringBuilder (human readable output)
      }
      output = net.rnnTimeStep(nextInput) //Do one time step of forward pass
    }
    val out = new mutable.ArrayBuffer[String](numSamples)
    for (i <- 0 until numSamples) {
      out(i) = sb(i).toString
    }
    out.toArray
  }

  /**
    * Given a probability distribution over discrete classes, sample from the distribution
    * and return the generated class index.
    *
    * @param distribution Probability distribution over classes. Must sum to 1.0
    */
  private def sampleFromDistribution(distribution: Array[Double], rng: Random): Int = {
    val d = rng.nextDouble
    val i = distribution
      .toIterator
      .scanLeft(0.0)({ case (acc, p) => acc + p })
      .drop(1)
      .indexWhere(_ >= d)
    if (i >= 0) {
      i
    } else {
      //Should never happen if distribution is a valid probability distribution
      throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + distribution.sum)
    }
  }

}
