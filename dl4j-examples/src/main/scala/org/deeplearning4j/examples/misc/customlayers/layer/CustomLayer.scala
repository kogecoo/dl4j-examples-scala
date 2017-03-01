package org.deeplearning4j.examples.misc.customlayers.layer

import org.deeplearning4j.nn.api.ParamInitializer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.activations.{Activation, IActivation}
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.api.Layer

import java.util

/**
  * Layer configuration class for the custom layer example
  *
  * @author Alex Black
  */
object CustomLayer {

  //Here's an implementation of a builder pattern, to allow us to easily configure the layer
  //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
  class Builder extends FeedForwardLayer.Builder[Builder] {

    var secondActivationFunction: IActivation = null

    /**
      * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
      *
      * @param f Second activation function for the layer
      */
    def secondActivationFunction(f: String): Builder = {
      secondActivationFunction(Activation.fromString(f))
    }

    /**
      * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
      *
      * @param f Second activation function for the layer
      */
    def secondActivationFunction(f: Activation): Builder = {
      this.secondActivationFunction = f.getActivationFunction
      this
    }

    @SuppressWarnings(Array("unchecked")) //To stop warnings about unchecked cast. Not required.
    def build[L <: org.deeplearning4j.nn.conf.layers.Layer](): L = new CustomLayer(this).asInstanceOf[L]

  }

}

class CustomLayer(builder: CustomLayer.Builder) extends FeedForwardLayer(builder) {
  //We need a no-arg constructor so we can deserialize the configuration from JSON or YAML format
  // Without this, you will likely get an exception like the following:
  //com.fasterxml.jackson.databind.JsonMappingException: No suitable constructor found for type [simple type, class org.deeplearning4j.examples.misc.customlayers.layer.CustomLayer]: can not instantiate from JSON object (missing default constructor or creator, or perhaps need to add/enable type information?)

  private var secondActivationFunction: IActivation = builder.secondActivationFunction

  def getSecondActivationFunction: IActivation = {
    //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
    secondActivationFunction
  }

  def setSecondActivationFunction(secondActivationFunction: IActivation) {
    //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
    this.secondActivationFunction = secondActivationFunction
  }

  def instantiate(conf: NeuralNetConfiguration, iterationListeners: util.Collection[IterationListener],
                  layerIndex: Int, layerParamsView: INDArray, initializeParams: Boolean): Layer = {
    //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
    // (i.e., a CustomLayerImpl instance)
    //For the most part, it's the same for each type of layer

    val myCustomLayer: CustomLayerImpl = new CustomLayerImpl(conf)
    myCustomLayer.setListeners(iterationListeners) //Set the iteration listeners, if any
    myCustomLayer.setIndex(layerIndex) //Integer index of the layer

    //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
    // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
    // (i.e., it's a "view" array in that it's a subset of a larger array)
    // This is a row vector, with length equal to the number of parameters in the layer
    myCustomLayer.setParamsViewArray(layerParamsView)

    //Initialize the layer parameters. For example,
    // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
    // are in turn a view of the 'layerParamsView' array.
    val paramTable: util.Map[String, INDArray] = initializer.init(conf, layerParamsView, initializeParams)
    myCustomLayer.setParamTable(paramTable)
    myCustomLayer.setConf(conf)
    myCustomLayer.asInstanceOf[Layer]
  }

  def initializer: ParamInitializer = {
    //This method returns the parameter initializer for this type of layer
    //In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
    //For more complex layers, you may need to implement a custom parameter initializer
    //See the various parameter initializers here:
    //https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params

    DefaultParamInitializer.getInstance
  }
}
