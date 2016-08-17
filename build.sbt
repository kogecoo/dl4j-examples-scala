lazy val root = project.in(file("."))
    .aggregate(dl4j_examples, dl4j_spark_examples, datavec_examples)
    .settings(name := "deeplearningj4j-examples")
    .settings(commonSettings:_*)

lazy val dl4j_examples = project.in(file("dl4j-examples"))
  .settings(name := "dl4j-examples")
  .settings(commonSettings:_*)

lazy val dl4j_spark_examples = project.in(file("dl4j-spark-examples"))
  .settings(name := "dl4j-spark-examples")
  .settings(commonSettings:_*)

lazy val datavec_examples = project.in(file("datavec-examples"))
  .settings(name := "datavec-examples")
  .settings(commonSettings:_*)

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
