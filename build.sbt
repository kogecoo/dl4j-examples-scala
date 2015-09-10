name := "deeplearning4j-examples-scala"

version := "1.0"

scalaVersion := "2.10.4"

lazy val root = project.in(file("."))

libraryDependencies ++= Seq(
  "commons-io" % "commons-io" % "2.4",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc2.2",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc2.2",
  "org.jblas" % "jblas" % "1.2.4",
  "org.nd4j" % "canova-nd4j-image" % "0.0.0.6",
  "org.nd4j" % "nd4j-jblas" % "0.4-rc2.2",
  "org.nd4j" % "nd4j-x86" % "0.4-rc2.2"
)

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"
