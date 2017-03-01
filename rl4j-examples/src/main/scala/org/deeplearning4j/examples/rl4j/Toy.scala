package org.deeplearning4j.examples.rl4j

import org.deeplearning4j.rl4j.learning.Learning
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.{AsyncNStepQLearningDiscrete, AsyncNStepQLearningDiscreteDense}
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense
import org.deeplearning4j.rl4j.mdp.toy.{HardDeteministicToy, SimpleToy, SimpleToyState}
import org.deeplearning4j.rl4j.network.dqn.{DQNFactoryStdDense, IDQN}
import org.deeplearning4j.rl4j.space.DiscreteSpace
import org.deeplearning4j.rl4j.util.DataManager

/**
  * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
  *
  *         main example for toy DQN
  *
  */
object Toy {

  var TOY_QL = new QLearning.QLConfiguration(
    123,    //Random seed
    100000, //Max step By epoch
    80000,  //Max step
    10000,  //Max size of experience replay
    32,     //size of batches
    100,    //target update (hard)
    0,      //num step noop warmup
    0.05,   //reward scaling
    0.99,   //gamma
    10.0,   //td-error clipping
    0.1f,   //min epsilon
    2000,   //num step for eps greedy anneal
    true    //double DQN
  )

  var TOY_ASYNC_QL = new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(123, //Random seed
    100000, //Max step By epoch
    80000,  //Max step
    8,      //Number of threads
    5,      //t_max
    100,    //target update (hard)
    0,      //num step noop warmup
    0.1,    //reward scaling
    0.99,   //gamma
    10.0,   //td-error clipping
    0.1f,   //min epsilon
    2000    //num step for eps greedy anneal
  )

  var TOY_NET: DQNFactoryStdDense.Configuration = new DQNFactoryStdDense.Configuration(
    3,     //number of layers
    16,    //number of hidden nodes
    0.001, //learning rate
    0.01   //l2 regularization
  )

  def main(args: Array[String]) {
    simpleToy()
    //toyAsyncNstep();
  }

  def simpleToy() {

    //record the training data in rl4j-data in a new folder
    val manager: DataManager = new DataManager

    //define the mdp from toy (toy length)
    val mdp: SimpleToy = new SimpleToy(20)

    //define the training method
    val dql: Learning[SimpleToyState, Integer, DiscreteSpace, IDQN] = new QLearningDiscreteDense[SimpleToyState](mdp, TOY_NET, TOY_QL, manager)

    //enable some logging for debug purposes on toy mdp
    mdp.setFetchable(dql)

    //start the training
    dql.train()

    //useless on toy but good practice!
    mdp.close()
  }

  def hardToy() {

    //record the training data in rl4j-data in a new folder
    val manager: DataManager = new DataManager

    //define the mdp from toy (toy length)
    val mdp = new HardDeteministicToy

    //define the training
    val dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager)

    //start the training
    dql.train()

    //useless on toy but good practice!
    mdp.close()
  }

  def toyAsyncNstep() {

    //record the training data in rl4j-data in a new folder
    val manager: DataManager = new DataManager

    //define the mdp
    val mdp: SimpleToy = new SimpleToy(20)

    //define the training
    val dql = new AsyncNStepQLearningDiscreteDense[SimpleToyState](mdp, TOY_NET, TOY_ASYNC_QL, manager)

    //enable some logging for debug purposes on toy mdp
    mdp.setFetchable(dql)

    //start the training
    dql.train()

    //useless on toy but good practice!
    mdp.close()
  }
}
