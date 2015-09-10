package org.deeplearning4j.examples.deepbelief

import org.apache.commons.io.FileUtils
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.conf.layers.{ OutputLayer, RBM }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{ DataSet, SplitTestAndTrain }
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import java.io._
import java.nio.file.{ Files, Paths }
import java.util.{ Arrays, Random }
import scala.collection.JavaConverters._

object DBNIrisExample {

    lazy val log = LoggerFactory.getLogger(DBNIrisExample.getClass)

    def main(args: Array[String]) = {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1

        val numRows = 4
        val numColumns = 1
        val outputNum = 3
        val numSamples = 150
        val batchSize = 150
        val iterations = 5
        val splitTrainNum = (batchSize * .8).toInt
        val seed = 123
        val listenerFreq = 1

        log.info("Load data....")
        val iter: DataSetIterator = new IrisDataSetIterator(batchSize, numSamples)
        val next: DataSet = iter.next()
        next.normalizeZeroMeanZeroUnitVariance()

        log.info("Split data....")
        val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed))
        val train: DataSet = testAndTrain.getTrain()
        val test: DataSet = testAndTrain.getTest()
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true

        log.info("Build model....")
        val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed) // Seed to lock in weight initialization for tuning
                .iterations(iterations) // # training iterations predict/classify & backprop
                .learningRate(1e-6f) // Optimization step size
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop method (calculate the gradients)
                .l1(1e-1).regularization(true).l2(2e-4)
                .useDropConnect(true)
                .list(2) // # NN layers (does not count input layer)
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                                .nIn(numRows * numColumns) // # input nodes
                                .nOut(3) // # fully connected hidden layer nodes. Add list if multiple layers.
                                .weightInit(WeightInit.XAVIER) // Weight initialization method
                                .k(1) // # contrastive divergence iterations
                                .activation("relu") // Activation function type
                                .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
                                .updater(Updater.ADAGRAD)
                                .dropOut(0.5)
                                .build()
                ) // NN layer type
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .nIn(3) // # input nodes
                                .nOut(outputNum) // # output nodes
                                .activation("softmax")
                                .build()
                ) // NN layer type
                .build()
        val model = new MultiLayerNetwork(conf)
        model.init()
//        model.setListeners(Arrays.asList(new ScoreIterationListener(listenerFreq),
//                new GradientPlotterIterationListener(listenerFreq),
//                new LossPlotterIterationListener(listenerFreq)))


        model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)
        log.info("Train model....")
        model.fit(train)

        log.info("Evaluate weights....")
        model.getLayers.foreach { case (layer: org.deeplearning4j.nn.api.Layer) =>
            val w: INDArray = layer.getParam(DefaultParamInitializer.WEIGHT_KEY)
            log.info("Weights: " + w)
        }

        log.info("Evaluate model....")
        val eval = new Evaluation(outputNum)
        val output = model.output(test.getFeatureMatrix())

        (0 until output.rows()).foreach { i =>
            val actual = train.getLabels().getRow(i).toString().trim()
            val predicted = output.getRow(i).toString().trim()
            log.info("actual " + actual + " vs predicted " + predicted)
        }

        eval.eval(test.getLabels(), output)
        log.info(eval.stats())
        log.info("****************Example finished********************")

        val fos: OutputStream = Files.newOutputStream(Paths.get("coefficients.bin"))
        val dos = new DataOutputStream(fos)
        Nd4j.write(model.params(), dos)
        dos.flush()
        dos.close()
        FileUtils.writeStringToFile(new File("conf.json"), model.getLayerWiseConfigurations().toJson())

        val confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")))
        val dis = new DataInputStream(new FileInputStream("coefficients.bin"))
        val newParams = Nd4j.read(dis)
        dis.close()
        val savedNetwork = new MultiLayerNetwork(confFromJson)
        savedNetwork.init()
        savedNetwork.setParameters(newParams)
        System.out.println("Original network params " + model.params())
        System.out.println(savedNetwork.params())



    }
}
