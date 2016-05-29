DL4J Release 0.4 Examples for Scala
=========================

 This project is a simple Scala porting of [deeplearning4j-0.4-example](https://github.com/deeplearning4j/dl4j-0.4-examples).

The main purpose of this project is:

* Show example of how to use deeplearning4j Java API from scala

* Provide a sbt project template depends on DL4J


Every examples in this demonstration directly calls Java APIs of DL4J.
If you want to write codes in the scala manner strongly,
you should consider to use [deeplearning4s](https://github.com/deeplearning4j/deeplearning4s).


### Before building this project
 Before you build this library, it needs to modify the line in build.sbt for specifying your OS environment.
The target line is

`"org.nd4j" % "nd4j-native" % "0.4-rc3.9",`

and you need to modify it to

`"org.nd4j" % "nd4j-native" % "0.4-rc3.9" classifier "" classifier "[OS ENVIRONMENT]",`

Note that [OS ENVIRONMENT] will be specified like:

* windows-x86_64
* linux-x86_64
* macosx-x86_64

For example, in OSX case, the line is:

* `"org.nd4j" % "nd4j-native" % "0.4-rc3.9" classifier "" classifier "macosx-x86_64",`

For more detailed information will be found in [Nd4j document page](http://nd4j.org/dependencies.html).

---

Repository of Deeplearning4J neural net examples:

- Convolutional Neural Nets
- Recurrent Neural Nets
- Deep-belief Neural Nets
- Restricted Boltzmann Machines
- Recursive Neural Nets
- TSNE
- Word2Vec & GloVe
- Anomaly Detection


---

## Documentation
For more information, check out [deeplearning4j.org](http://deeplearning4j.org/) and its [JavaDoc](http://deeplearning4j.org/doc/).

If you notice issues, please log them, and if you want to contribute, submit a pull request. Input is welcome here.

