package org.deeplearning4j.examples.misc.modelsaving

import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

/**
  * A very simple example for saving and loading a ComputationGraph
  *
  * @author Alex Black
  */
object SaveLoadComputationGraph {
  @throws[Exception]
  def main(args: Array[String]) {
    //Define a simple ComputationGraph:
    val conf = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .learningRate(0.1)
      .graphBuilder
      .addInputs("in")
      .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build, "in")
      .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build, "layer0")
      .setOutputs("layer1")
      .backprop(true).pretrain(false).build

    val net = new ComputationGraph(conf)
    net.init()

    //Save the model
    val locationToSave = new File("MyComputationGraph.zip") //Where to save the network. Note: the file is in .zip format - can be opened externally
    val saveUpdater = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
    ModelSerializer.writeModel(net, locationToSave, saveUpdater)

    //Load the model
    val restored = ModelSerializer.restoreComputationGraph(locationToSave)

    println("Saved and loaded parameters are equal:      " + net.params == restored.params)
    println("Saved and loaded configurations are equal:  " + net.getConfiguration == restored.getConfiguration)
  }
}
