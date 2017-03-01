package org.deeplearning4j.examples.rl4j

import org.deeplearning4j.rl4j.learning.async.a3c.discrete.{A3CDiscrete, A3CDiscreteDense}
import org.deeplearning4j.rl4j.mdp.gym.GymEnv
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense
import org.deeplearning4j.rl4j.space.{Box, DiscreteSpace}
import org.deeplearning4j.rl4j.util.DataManager

/**
  * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
  *
  *         main example for A3C on cartpole
  *
  */
object A3CCartpole {
  private val CARTPOLE_A3C = new A3CDiscrete.A3CConfiguration(
    123,    //Random seed
    200,    //Max step By epoch
    500000, //Max step
    16,     //Number of threads
    5,      //t_max
    10,     //num step noop warmup
    0.01,   //reward scaling
    0.99,   //gamma
    10.0    //td-error clipping
  )

  private val CARTPOLE_NET_A3C = new ActorCriticFactorySeparateStdDense.Configuration(
    3,     //number of layers
    16,    //number of hidden nodes
    0.001, //learning rate
    0.000  //l2 regularization
  )

  def main(args: Array[String]) {
    A3CcartPole()
  }

  def A3CcartPole() {
    //record the training data in rl4j-data in a new folder
    val manager: DataManager = new DataManager(true)

    //define the mdp from gym (name, render)
    var mdp: GymEnv[Box, Integer, DiscreteSpace] = null
    try {
      mdp = new GymEnv("CartPole-v0", false, false)
    } catch { case e: RuntimeException =>
      System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.")
    }

    //define the training
    val dql = new A3CDiscreteDense[Box](mdp, CARTPOLE_NET_A3C, CARTPOLE_A3C, manager)
    //start the training
    dql.train()
    //close the mdp (http connection)
    mdp.close()
  }
}
