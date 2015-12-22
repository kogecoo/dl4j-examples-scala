name := "deeplearning4j-examples-scala"

version := "1.0"

scalaVersion := "2.10.4"

lazy val root = project.in(file("."))

val nd4jVersion = "0.4-rc3.7"
val dl4jVersion =	"0.4-rc3.7"
val canovaVersion = "0.0.0.13"
val jacksonVersion = "2.5.1"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-ui" % dl4jVersion,
  "org.nd4j" % "canova-nd4j-image" % canovaVersion,
  "org.nd4j" % "canova-nd4j-codec" % canovaVersion,
  "org.nd4j" % "nd4j-x86" % nd4jVersion
)

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"
