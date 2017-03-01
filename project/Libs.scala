package dl4j_examples_scala.builds

import sbt._

object Libs {

  val dl4jVer = "0.7.2"

  val arbiterDl4j         = "org.deeplearning4j" %  "arbiter-deeplearning4j"          % dl4jVer
  val datavecApi          = "org.datavec"        %  "datavec-api"                     % dl4jVer
  val datavecDataCodec    = "org.datavec"        %  "datavec-data-codec"              % dl4jVer
  val datavecSpark        = "org.datavec"        %% "datavec-spark"                   % dl4jVer
  val dl4jCore            = "org.deeplearning4j" %  "rl4j-core"                       % dl4jVer
  val dl4jNlp             = "org.deeplearning4j" %  "deeplearning4j-nlp"              % dl4jVer
  val dl4jParallelWrapper = "org.deeplearning4j" %  "deeplearning4j-parallel-wrapper" % dl4jVer
  val dl4jSpark           = "org.deeplearning4j" %% "dl4j-spark"                      % dl4jVer
  val dl4jUi              = "org.deeplearning4j" %% "deeplearning4j-ui"               % dl4jVer
  val nd4jCuda75          = "org.nd4j"           %  "nd4j-cuda-7.5"                   % dl4jVer
  val nd4jCuda75Platform  = "org.nd4j"           %  "nd4j-cuda-7.5-platform"          % dl4jVer
  val nd4jCuda80Platform  = "org.nd4j"           %  "nd4j-cuda-8.0-platform"          % dl4jVer
  val nd4jNative          = "org.nd4j"           %  "nd4j-native"                     % dl4jVer
  val nd4jNativePlatform  = "org.nd4j"           %  "nd4j-native-platform"            % dl4jVer
  val rl4jGym             = "org.deeplearning4j" %  "rl4j-gym"                        % dl4jVer


  val guava       = "com.google.guava"          %  "guava"           % "19.0"
  val httpClient  = "org.apache.httpcomponents" %  "httpclient"      % "4.3.5"
  val imageIOCore = "com.twelvemonkeys.imageio" % "imageio-core"     % "3.3.2"
  val jCommander  = "com.beust"                 %  "jcommander"      % "1.27"
  val jcommon     = "org.jfree"                 %  "jcommon"         % "1.0.23"
  val jfreeChart  = "jfree"                     %  "jfreechart"      % "1.0.13"
  val logback     = "ch.qos.logback"            %  "logback-classic" % "1.1.7"
  val sparkCore   = "org.apache.spark"          %% "spark-core"      % "1.6.0" exclude("javax.servlet", "servlet-api")
}
