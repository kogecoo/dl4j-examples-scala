name := "deeplearning4j-examples-scala"

version := "1.0"

scalaVersion := "2.10.4"

lazy val root = project.in(file("."))

libraryDependencies ++= Seq(
  "org.apache.zookeeper" % "zookeeper" % "3.3.2",
  "org.nd4j" % "nd4j-jblas" % "0.4-rc0",
  "org.nd4j" % "nd4j-x86" % "0.4-rc0",
  "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc0",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc0",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc0"
)

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"
