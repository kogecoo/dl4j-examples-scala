Deeplearning4J Examples for Scala
=========================

 This project is a simple Scala porting of [deeplearning4j-example](https://github.com/deeplearning4j/dl4j-examples).

The main purpose of this project is:

* Show example of how to use deeplearning4j Java API from scala

* Provide a sbt project template depends on DL4J

 Every examples in this demonstration directly calls Java APIs of DL4J.
If you strongly want to write codes in the scala manner,
you should consider to use [deeplearning4s](https://github.com/deeplearning4j/deeplearning4s).

 Note that currently `dl4j-spark-examples` are not merged yet.

### Getting Started

 This repository has some projects.
To choose and run a project, followings commands are useful.

* arbiter examples
  * `sbt "; project arbiter-examples; run"`

* datavec examples
  * `sbt "; project datavec-examples; run"`

* dl4j cuda specific examples
  * `sbt "; project dl4j-cuda-specific-examples; run"`

* dl4j examples
  * `sbt "; project dl4j-examples; run"`

* dl4j spark examples
  * `sbt "; project dl4j-spark; run"`

* nd4j examples
  * `sbt "; project nd4j-examples; run"`

* rl4j examples
  * `sbt "; project rl4j-examples; run"`

### Switch nd4j backend
To switch nd4j backend, set one of following options to `nd4jBackend` variable in `build.sbt`.
* Libs.nd4jNativePlatform (default)
* Libs.nd4jCuda75Platform
* Libs.nd4jCuda80Platform

---

Repository of Deeplearning4J neural net examples:

- MLP Neural Nets
- Convolutional Neural Nets
- Recurrent Neural Nets
- TSNE
- Word2Vec & GloVe
- Anomaly Detection

---

## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.


