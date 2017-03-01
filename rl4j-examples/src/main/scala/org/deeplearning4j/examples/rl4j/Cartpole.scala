package org.deeplearning4j.examples.rl4j

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense
import org.deeplearning4j.rl4j.mdp.gym.GymEnv
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense
import org.deeplearning4j.rl4j.policy.DQNPolicy
import org.deeplearning4j.rl4j.space.{Box, DiscreteSpace}
import org.deeplearning4j.rl4j.util.DataManager

import java.util.logging.Logger

/**
  * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
  *
  *         Main example for Cartpole DQN
  *
  ***/
object Cartpole {
  var CARTPOLE_QL: QLearning.QLConfiguration = new QLearning.QLConfiguration(
    123,    //Random seed
    200,    //Max step By epoch
    150000, //Max step
    150000, //Max size of experience replay
    32,     //size of batches
    500,    //target update (hard)
    10,     //num step noop warmup
    0.01,   //reward scaling
    0.99,   //gamma
    1.0,    //td-error clipping
    0.1f,   //min epsilon
    1000,   //num step for eps greedy anneal
    true    //double DQN
  )

  var CARTPOLE_NET: DQNFactoryStdDense.Configuration = new DQNFactoryStdDense.Configuration(
    3,     //number of layers
    16,    //number of hidden nodes
    0.001, //learning rate
    0.00   //l2 regularization
  )

  def main(args: Array[String]) {
    cartPole()
    loadCartpole()
  }

  def cartPole() {
    //record the training data in rl4j-data in a new folder (save)
    val manager: DataManager = new DataManager(true)
    //define the mdp from gym (name, render)
    var mdp: GymEnv[Box, Integer, DiscreteSpace] = null
    try {
      mdp = new GymEnv("CartPole-v0", false, false)
    }
    catch { case e: RuntimeException =>
      print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.")
    }
    //define the training
    val dql = new QLearningDiscreteDense[Box](mdp, CARTPOLE_NET, CARTPOLE_QL, manager)

    //train
    dql.train()

    //get the final policy
    val pol: DQNPolicy[Box] = dql.getPolicy

    //serialize and save (serialization showcase, but not required)
    pol.save("/tmp/pol1")

    //close the mdp (close http)
    mdp.close()
  }

  def loadCartpole() {
    //showcase serialization by using the trained agent on a new similar mdp (but render it this time)
    //define the mdp from gym (name, render)
    val mdp2 = new GymEnv[Box, Integer, DiscreteSpace]("CartPole-v0", true, false)

    //load the previous agent
    val pol2: DQNPolicy[Box] = DQNPolicy.load("/tmp/pol1")

    //evaluate the agent
    var rewards: Double = 0
    for (i <- 0 until 1000) {
      mdp2.reset
      val reward: Double = pol2.play(mdp2)
      rewards += reward
      Logger.getAnonymousLogger.info("Reward: " + reward)
    }
    Logger.getAnonymousLogger.info("average: " + rewards / 1000)
  }

}
