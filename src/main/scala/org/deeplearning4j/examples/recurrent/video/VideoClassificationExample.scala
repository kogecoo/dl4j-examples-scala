package org.deeplearning4j.examples.recurrent.video

import java.io.File

import org.apache.commons.io.FilenameUtils
import org.apache.uima.util.FileUtils
import org.canova.api.conf.Configuration
import org.canova.api.records.reader.SequenceRecordReader
import org.canova.api.records.reader.impl.CSVSequenceRecordReader
import org.canova.api.split.NumberedFileInputSplit
import org.canova.codec.reader.CodecRecordReader
import org.deeplearning4j.datasets.canova.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.{AsyncDataSetIterator, DataSetIterator}
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.preprocessor._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.dataset.api
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConverters._

/**
 * Example: Combine convolutional, max pooling, dense (feed forward) and recurrent (LSTM) layers to classify each
 * frame of a video (using a generated/synthetic video data set)
 * Specifically, each video contains a shape (randomly selected: circles, squares, lines, arcs) which persist for
 * multiple frames (though move between frames) and may leave the frame. Each video contains multiple shapes which
 * are shown for some random number of frames.
 * The network needs to classify these shapes, even when the shape has left the frame.
 *
 * This example is somewhat contrived, but shows data import and network configuration for classifying video frames.
 *
 * *******************************************************
 * WARNING: THIS EXAMPLE GENERATES A LARGE DATA SET
 * This examples does NOT automatically delete this data set after the example is complete.
 * *******************************************************
  *
  * @author Alex Black
 */
object VideoClassificationExample {

    final val N_VIDEOS_TO_GENERATE = 500
    final val V_WIDTH = 130
    final val V_HEIGHT = 130
    final val V_NFRAMES = 150

    def main(args: Array[String]) = {

        val miniBatchSize = 10
        val needGenerateData = true

        val tempDir = System.getProperty("java.io.tmpdir")
        val dataDirectory = FilenameUtils.concat(tempDir, "DL4JVideoShapesExample/");   //Location to store generated data set

        //Generate data: number of .mp4 videos for input, plus .txt files for the labels
        if (needGenerateData) {
            System.out.println("Starting data generation...")
            generateData(dataDirectory)
            System.out.println("Data generation complete")
        }

        //Set up network architecture:
        val updater = Updater.ADAGRAD
        val conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .regularization(true).l2(0.001) //l2 regularization on all layers
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.04)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(3) //3 channels: RGB
                        .nOut(30)
                        .stride(4, 4)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .build())   //Output: (130-10+0)/4+1 = 31 -> 31*31*30
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2).build())   //(31-3+0)/2+1 = 15
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(30)
                        .nOut(10)
                        .stride(2, 2)
                        .activation("relu")
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .build())   //Output: (15-3+0)/2+1 = 7 -> 7*7*10 = 490
                .layer(3, new DenseLayer.Builder()
                        .activation("relu")
                        .nIn(490)
                        .nOut(50)
                        .weightInit(WeightInit.RELU)
                        .updater(updater)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(0.01)
                        .build())
                .layer(4, new GravesLSTM.Builder()
                        .activation("softsign")
                        .nIn(50)
                        .nOut(50)
                        .weightInit(WeightInit.XAVIER)
                        .updater(updater)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .learningRate(0.008)
                        .build())
                .layer(5, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(50)
                        .nOut(4)    //4 possible shapes: circle, square, arc, line
                        .updater(updater)
                        .weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, 3))
                .inputPreProcessor(3, new CnnToFeedForwardPreProcessor(7, 7, 10))
                .inputPreProcessor(4, new FeedForwardToRnnPreProcessor())
                .pretrain(false).backprop(true)
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(V_NFRAMES / 5)
                .tBPTTBackwardLength(V_NFRAMES / 5)
                .build()

        val net = new MultiLayerNetwork(conf)
        net.init()
        net.setListeners(new ScoreIterationListener(1))

        System.out.println("Number of parameters in network: " + net.numParams())
        (0 until net.getnLayers()).foreach { i =>
            System.out.println("Layer " + i + " nParams = " + net.getLayer(i).numParams())
        }

        val testStartIdx = (0.9 * N_VIDEOS_TO_GENERATE).asInstanceOf[Int]  //90% in train, 10% in test
        val nTest = N_VIDEOS_TO_GENERATE - testStartIdx

        //Conduct learning
        System.out.println("Starting training...")
        val nTrainEpochs = 15
        (0 until nTrainEpochs).foreach { i =>
            val trainData: DataSetIterator = getDataSetIterator(dataDirectory, 0, testStartIdx - 1, miniBatchSize)
            while(trainData.hasNext)
                net.fit(trainData.next())
            Nd4j.saveBinary(net.params(),new File("videomodel.bin"))
            FileUtils.saveString2File(conf.toJson,new File("videoconf.json"))
            System.out.println("Epoch " + i + " complete")

            //Evaluate classification performance:
            evaluatePerformance(net,testStartIdx,nTest,dataDirectory)
        }
    }

    private[this] def generateData(path: String): Unit = {
        val f = new File(path)
        if (!f.exists()) f.mkdir()

        /** The data generation code does support the addition of background noise and distractor shapes (shapes which
         * are shown for one frame only in addition to the target shape) but these are disabled by default.
         * These can be enabled to increase the complexity of the learning task.
         */
        VideoGenerator.generateVideoData(path, "shapes", N_VIDEOS_TO_GENERATE,
                V_NFRAMES, V_WIDTH, V_HEIGHT,
                3,      //Number of shapes per video. Switches from one shape to another randomly over time
                false,   //Background noise. Significantly increases video file size
                0,      //Number of distractors per frame ('distractors' are shapes show for one frame only)
                12345L);    //Seed, for reproducability when generating data
    }

    private[this] def evaluatePerformance(net: MultiLayerNetwork, testStartIdx: Int, nExamples: Int, outputDirectory: String) {
        //Assuming here that the full test data set doesn't fit in memory -> load 10 examples at a time
        val labelMap = Map[Int, String](
            0 -> "circle",
            1 -> "square",
            2 -> "arc",
            3 -> "line"
        )
        val forConstructor = labelMap.map({ case (k, v) => (new Integer(k), v) }).asJava
        val evaluation = new Evaluation(forConstructor)

        val testData = getDataSetIterator(outputDirectory, testStartIdx, nExamples, 10)
        while(testData.hasNext) {
            val dsTest = testData.next()
            val predicted = net.output(dsTest.getFeatureMatrix, false)
            val actual = dsTest.getLabels
            evaluation.evalTimeSeries(predicted, actual)
        }

        System.out.println(evaluation.stats())
    }

    private[this] def getDataSetIterator(dataDirectory: String, startIdx: Int, nExamples: Int, miniBatchSize: Int): DataSetIterator = {
        //Here, our data and labels are in separate files
        //videos: shapes_0.mp4, shapes_1.mp4, etc
        //labels: shapes_0.txt, shapes_1.txt, etc. One time step per line

        val featuresTrain: SequenceRecordReader = getFeaturesReader(dataDirectory, startIdx, nExamples)
        val labelsTrain: SequenceRecordReader = getLabelsReader(dataDirectory, startIdx, nExamples)

        val sequenceIter = new SequenceRecordReaderDataSetIterator(featuresTrain, labelsTrain, miniBatchSize, 4, false)
        sequenceIter.setPreProcessor(new VideoPreProcessor())

        //AsyncDataSetIterator: Used to (pre-load) load data in a separate thread
        new AsyncDataSetIterator(sequenceIter,1)
    }

    private[this] def getFeaturesReader(path: String, startIdx: Int, num: Int): SequenceRecordReader = {
        //InputSplit is used here to define what the file paths look like
        val is = new NumberedFileInputSplit(path + "shapes_%d.mp4", startIdx, startIdx + num - 1)

        val conf = new Configuration()
        conf.set(CodecRecordReader.RAVEL, "true")
        conf.set(CodecRecordReader.START_FRAME, "0")
        conf.set(CodecRecordReader.TOTAL_FRAMES, String.valueOf(V_NFRAMES))
        conf.set(CodecRecordReader.ROWS, String.valueOf(V_WIDTH))
        conf.set(CodecRecordReader.COLUMNS, String.valueOf(V_HEIGHT))
        val crr = new CodecRecordReader()
        crr.initialize(conf, is)
        crr
    }

    private[this] def getLabelsReader(path: String, startIdx: Int, num: Int): SequenceRecordReader = {
        val isLabels = new NumberedFileInputSplit(path + "shapes_%d.txt", startIdx, startIdx + num - 1)
        val csvSeq = new CSVSequenceRecordReader()
        csvSeq.initialize(isLabels)
        csvSeq
    }

    private[this] class VideoPreProcessor extends DataSetPreProcessor {
        def preProcess(toPreProcess: api.DataSet): Unit = {
            toPreProcess.getFeatureMatrix.divi(255) //[0,255] -> [0,1] for input pixel values
        }
    }

}