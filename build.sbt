import dl4j_examples_scala.builds.Libs

lazy val nd4jBackend = Libs.nd4jNativePlatform
// or Libs.nd4jCuda75Platform
// or Libs.nd4jCuda80Platform

lazy val root = project.in(file("."))
    .aggregate(
      `arbiter-examples`,
      `datavec-examples`,
      `dl4j-cuda-specific-examples`,
      `dl4j-examples`,
      //`dl4j-spark-examples`,
      `dl4j-spark`,
      `nd4j-examples`,
      `rl4j-examples`
    )
    .settings(name := "deeplearningj4j-examples")
    .settings(commonSettings:_*)

lazy val `arbiter-examples` = project.in(file("arbiter-examples"))
  .settings(name := "arbiter-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.arbiterDl4j,
    Libs.dl4jCore,
    Libs.guava,
    nd4jBackend
  ))

lazy val `datavec-examples` = project.in(file("datavec-examples"))
  .settings(name := "datavec-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.datavecApi,
    Libs.datavecSpark
  ))

lazy val `dl4j-cuda-specific-examples` = project.in(file("dl4j-cuda-specific-examples"))
  .settings(name := "dl4j-cuda-specific-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.guava,
    Libs.dl4jCore,
    Libs.dl4jNlp,
    Libs.dl4jParallelWrapper,
    Libs.dl4jUi,
    Libs.nd4jCuda75Platform
  ))

lazy val `dl4j-examples` = project.in(file("dl4j-examples"))
  .settings(name := "dl4j-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.datavecDataCodec,
    Libs.guava,
    Libs.imageIOCore,
    Libs.jfreeChart,
    Libs.jcommon,
    Libs.dl4jCore,
    Libs.dl4jNlp,
    Libs.dl4jUi,
    Libs.httpClient,
    nd4jBackend
  ))

/*
lazy val `dl4j-spark-examples` = project.in(file("dl4j-spark-examples"))
  .settings(name := "dl4j-spark-examples")
  .settings(commonSettings:_*)
  .aggregate(`dl4j-spark`)
*/


lazy val `dl4j-spark` = project.in(file("dl4j-spark-examples/dl4j-spark"))
  .settings(name := "dl4j-spark")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.dl4jSpark,
    Libs.jCommander,
    Libs.sparkCore,
    nd4jBackend
  ))

lazy val `nd4j-examples` = project.in(file("nd4j-examples"))
  .settings(name := "nd4j-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.logback,
    nd4jBackend
  ))


lazy val `rl4j-examples` = project.in(file("rl4j-examples"))
  .settings(name := "rl4j-examples")
  .settings(commonSettings:_*)
  .settings(libraryDependencies ++= Seq(
    Libs.dl4jCore,
    Libs.rl4jGym,
    nd4jBackend
  ))


lazy val commonSettings = Seq(
  version := "1.0",
  scalaVersion := "2.10.6",
  classpathTypes += "maven-plugin",
  resolvers ++= commonResolvers
)

lazy val commonResolvers = Seq(
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/",
  "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"
)


