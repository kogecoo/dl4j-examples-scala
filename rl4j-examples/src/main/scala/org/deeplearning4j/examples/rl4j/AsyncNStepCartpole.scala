package org.deeplearning4j.examples.rl4j

import org.deeplearning4j.rl4j.space.{Box, DiscreteSpace}
import org.deeplearning4j.rl4j.learning.async.AsyncLearning
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense
import org.deeplearning4j.rl4j.mdp.gym.GymEnv
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.util.DataManager

/**
  * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
  *
  *         main example for Async NStep QLearning on cartpole
  */
object AsyncNStepCartpole {

  var CARTPOLE_NSTEP = new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
    123,    //Random seed
    200,    //Max step By epoch
    300000, //Max step
    16,     //Number of threads
    5,      //t_max
    100,    //target update (hard)
    10,     //num step noop warmup
    0.01,   //reward scaling
    0.99,   //gamma
    100.0,  //td-error clipping
    0.1f,   //min epsilon
    9000    //num step for eps greedy anneal)
  )

  var CARTPOLE_NET_NSTEP = new DQNFactoryStdDense.Configuration(
    3,      //number of layers
    16,     //number of hidden nodes
    0.0005, //learning rate
    0.001   //l2 regularization
  )

  def main(args: Array[String]) {
    cartPole()
  }

  def cartPole() {
    //record the training data in rl4j-data in a new folder
    val manager: DataManager = new DataManager(true)

    //define the mdp from gym (name, render)
    var mdp: GymEnv[Box, Integer, DiscreteSpace] = null
    try {
      mdp = new GymEnv("CartPole-v0", false, false)
    } catch { case e: RuntimeException =>
      print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.")
    }

    //define the training
    val dql = new AsyncNStepQLearningDiscreteDense[Box](mdp, CARTPOLE_NET_NSTEP, CARTPOLE_NSTEP, manager)
    //train
    dql.train()
    //close the mdp (close connection)
    mdp.close()
  }
}
