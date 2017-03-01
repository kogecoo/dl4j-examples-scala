package org.deeplearning4j.examples.tictactoe

import java.io.{File, FileWriter}
import java.util
import java.util.Date

import org.datavec.api.util.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._

/**
  * This program generates basic data to be used in Training Program.
  * It performs following major steps
  * - generates all possible game states
  * - reward all game states generated in above step by finding winning state, assign it to value 1 and goes back upto first step through all steps and
  * calculates probability of each step in the game to make that move win game in the last state.
  * - Writes all states data along with probability of each state to win the game which was calculated in above step.
  * Note :
  * - Used <b>http://www.se16.info/hgb/tictactoe.htm</b> link to understand all possible number of moves in Tic-Tac-Toe game.
  * - Refer ReadMe.txt for detail explanation of each step.
  * <p>
  * <b>Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com), 19-Jan-2017.</b>
  */
object TicTacToeData {
  /**
    * Main function that calls all major functions one-by-one to generate training data to be used in training program.
    */
  @throws[Exception]
  def main(args: Array[String]) {
    val filePath = new ClassPathResource("tictactoe").getFile.getAbsolutePath + File.separator + "AllMoveWithReward.txt"
    val data = new TicTacToeData
    println("Data Processing Started : " + (new Date).toString)
    data.generatePossibleGames()
    println("All possible game state sequence generated, Finished At : " + (new Date).toString)
    data.rewardGameState()
    println("Reward calculation finished : " + (new Date).toString)
    data.writeFinalData(filePath)
    println("File generation completed : " + (new Date).toString)
  }
}

class TicTacToeData {
  //All these private variables are not meant to be used from outside of the class. So, no getter/setter methods are provided.
  private val moveSequenceList = new util.ArrayList[INDArray]
  private val oddPlayerWiningList = new util.ArrayList[INDArray]
  private val evenPlayerWiningList = new util.ArrayList[INDArray]
  private val middleList = new util.ArrayList[INDArray]
  private val finalOutputArrayList = new util.ArrayList[INDArray]
  private val finalProbabilityValueList = new util.ArrayList[Double]
  private var previousMoveNumber = 0

  /**
    * Initiate generating all possible game states. Refer ReadMe.txt for detailed explanation.
    */
  def generatePossibleGames() {
    try {
      for (index <- 1 to 9) {
        generateStateBasedOnMoveNumber(index)
      }
    } catch { case e: Exception =>
        println(e.toString)
    }
    /*Here  process odd and Draw using odd list*/ oddPlayerWiningList.addAll(moveSequenceList)
  }

  /**
    * This function allocates reward points to each state of the game based on the winning state.
    * For all elements in oddPlayerWiningList, evenPlayerWiningList and middleList (which contains intermediate entries before winning or draw).
    * Refer ReadMe.txt for detailed explanation.
    */
  def rewardGameState() {
    oddPlayerWiningList.asScala.foreach { a =>
      generateGameStatesAndRewardToIt(a, 0) //0 odd for position  and 1 for even Position
    }
    evenPlayerWiningList.asScala.foreach { a =>
      generateGameStatesAndRewardToIt(a, 1)
    }
    middleList.asScala.foreach { element =>
      addToFinalOutputList(element, 0.5)
    }
  }

  /**
    * This function called by generatePossibleGames. It is the main function that generates all possible game states.
    * Refer ReadMe.txt for detailed explanation.
    */
  @throws[Exception]
  private def generateStateBasedOnMoveNumber(moveNumber: Int) {
    val newMoveNumber: Int = previousMoveNumber + 1
    if (newMoveNumber != moveNumber) {
      throw new Exception("Missing one or more moves between 1 to 9")
    } else if (moveNumber > 9 || moveNumber < 1) {
      throw new Exception("Invalid move number")
    }
    previousMoveNumber = newMoveNumber
    val tempMoveSequenceList: util.List[INDArray] = new util.ArrayList[INDArray]
    tempMoveSequenceList.addAll(moveSequenceList)
    moveSequenceList.clear()
    if (moveNumber == 1) {
      for (i <- 0 until 9) {
        moveSequenceList.add(Nd4j.ones(1, 9))
      }
    } else {
      val isOddMoveNumber = moveNumber % 2 != 0
      for (i <- tempMoveSequenceList.asScala.indices) {
        val moveArraySequence: INDArray = tempMoveSequenceList.get(i)
        for (j <- 0 until 9) {
          val temp1: INDArray = Nd4j.zeros(1, 9)
          Nd4j.copy(moveArraySequence, temp1)
          if (moveArraySequence.getInt(j) == 0) {
            temp1.putScalar(Array[Int](0, j), moveNumber)
            if (moveNumber > 4) {
              if (checkWin(temp1, isOddMoveNumber)) {
                if (isOddMoveNumber) {
                  oddPlayerWiningList.add(temp1)
                } else {
                  evenPlayerWiningList.add(temp1)
                }
              } else {
                moveSequenceList.add(temp1)
              }
            } else {
              moveSequenceList.add(temp1)
            }
          }
        }
      }
    }
  }

  /**
    * Identify the game state win/Draw.
    */
  private def checkWin(sequence: INDArray, isOdd: Boolean): Boolean = {
    val boardPosition1 = sequence.getDouble(0)
    val boardPosition2 = sequence.getDouble(1)
    val boardPosition3 = sequence.getDouble(2)
    val boardPosition4 = sequence.getDouble(3)
    val boardPosition5 = sequence.getDouble(4)
    val boardPosition6 = sequence.getDouble(5)
    val boardPosition7 = sequence.getDouble(6)
    val boardPosition8 = sequence.getDouble(7)
    val boardPosition9 = sequence.getDouble(8)

    val position1 = if (isOdd) { sequence.getDouble(0) % 2.0 != 0 }
      else { sequence.getDouble(0) % 2.0 == 0 }
    val position2 = if (isOdd) { sequence.getDouble(1) % 2.0 != 0 }
      else { sequence.getDouble(1) % 2.0 == 0 }
    val position3 = if (isOdd) { sequence.getDouble(2) % 2.0 != 0 }
      else { sequence.getDouble(2) % 2.0 == 0 }
    val position4 = if (isOdd) { sequence.getDouble(3) % 2.0 != 0 }
      else { sequence.getDouble(3) % 2.0 == 0 }
    val position5 = if (isOdd) { sequence.getDouble(4) % 2.0 != 0 }
      else { sequence.getDouble(4) % 2.0 == 0 }
    val position6 = if (isOdd) { sequence.getDouble(5) % 2.0 != 0 }
      else { sequence.getDouble(5) % 2.0 == 0 }
    val position7 = if (isOdd) { sequence.getDouble(6) % 2.0 != 0 }
      else { sequence.getDouble(6) % 2.0 == 0 }
    val position8 = if (isOdd) { sequence.getDouble(7) % 2.0 != 0 }
      else { sequence.getDouble(7) % 2.0 == 0 }
    val position9 = if (isOdd) { sequence.getDouble(8) % 2.0 != 0 }
      else (sequence.getDouble(8) % 2.0 == 0)
    if (((position1 && position2 && position3) && (boardPosition1 != 0 && boardPosition2 != 0 && boardPosition3 != 0)) ||
        ((position4 && position5 && position6) && (boardPosition4 != 0 && boardPosition5 != 0 && boardPosition6 != 0)) ||
        ((position7 && position8 && position9) && (boardPosition7 != 0 && boardPosition8 != 0 && boardPosition9 != 0)) ||
        ((position1 && position4 && position7) && (boardPosition1 != 0 && boardPosition4 != 0 && boardPosition7 != 0)) ||
        ((position2 && position5 && position8) && (boardPosition2 != 0 && boardPosition5 != 0 && boardPosition8 != 0)) ||
        ((position3 && position6 && position9) && (boardPosition3 != 0 && boardPosition6 != 0 && boardPosition9 != 0)) ||
        ((position1 && position5 && position9) && (boardPosition1 != 0 && boardPosition5 != 0 && boardPosition9 != 0)) ||
        ((position3 && position5 && position7) && (boardPosition3 != 0 && boardPosition5 != 0 && boardPosition7 != 0))) {

       true
    } else {
      false
    }
  }

  /**
    * This function generate all intermediate (including winning) game state from the winning state available oddPlayerWiningList or evenPlayerWiningList
    * and pass it to calculateReward function to calculate probability of all states of winning game.
    * Refer ReadMe.txt for detailed explanation.
    */
  private def generateGameStatesAndRewardToIt(output: INDArray, moveType: Int) {

    val maxArray: INDArray = Nd4j.max(output)
    val maxNumber: Double = maxArray.getDouble(0)

    val sequenceList: util.List[INDArray] = new util.ArrayList[INDArray]
    val sequenceArray: INDArray = Nd4j.zeros(1, 9)

    var move: Int = 1
    var positionOfDigit: Int = 0
    for (i <- 1 to maxNumber.toInt) {
      val newTempArray: INDArray = Nd4j.zeros(1, 9)
      positionOfDigit = getPosition(output, i)

      if (i % 2 == moveType) {
        Nd4j.copy(sequenceArray, newTempArray)
        sequenceList.add(newTempArray)
      } else {
        Nd4j.copy(sequenceArray, newTempArray)
        middleList.add(newTempArray)
      }
      sequenceArray.putScalar(Array[Int](0, positionOfDigit), move)
      move = move * (-1)
    }
    move = move * (-1)
    val newTempArray2: INDArray = Nd4j.zeros(1, 9)

    sequenceArray.putScalar(Array[Int](0, positionOfDigit), move)
    Nd4j.copy(sequenceArray, newTempArray2)
    sequenceList.add(newTempArray2)
    calculateReward(sequenceList)
  }

  /**
    * This function gives cell number of a particular move
    */
  private def getPosition(array: INDArray, number: Double): Int = {
    for (i <- 0 until array.length) {
      if (array.getDouble(i) == number) {
        return i
      }
    }
    0
  }

  /**
    * Function to calculate Temporal Difference. Refer ReadMe.txt for detailed explanation.
    */
  private def calculateReward(arrayList: util.List[INDArray]) {
    var probabilityValue: Double = 0
    for (p <- (0 until arrayList.size).reverse) {
      if (p == (arrayList.size - 1)) {
        probabilityValue = 1.0
      } else {
        probabilityValue = 0.5 + 0.1 * (probabilityValue - 0.5)
      }
      val stateAsINDArray: INDArray = arrayList.get(p)
      addToFinalOutputList(stateAsINDArray, probabilityValue)
    }
  }

  /**
    * This function adds game states to final list after calculating reward for each state of a winning game.
    */
  private def addToFinalOutputList(inputLabelArray: INDArray, inputRewardValue: Double) {
    val indexPosition: Int = finalOutputArrayList.indexOf(inputLabelArray)

    if (indexPosition != -1) {
      val rewardValue: Double = finalProbabilityValueList.get(indexPosition)
      val newUpdatedRewardValue: Double = if (rewardValue > inputRewardValue) rewardValue
      else inputRewardValue
      finalProbabilityValueList.set(indexPosition, newUpdatedRewardValue)
    } else {
      finalOutputArrayList.add(inputLabelArray)
      finalProbabilityValueList.add(inputRewardValue)
    }
  }

  /**
    * This function writes all states of all games into file along with their probability values.
    */
  def writeFinalData(saveFilePath: String) {
    try {
      val writer = new FileWriter(saveFilePath)
      try {
        val finalStringListForFile = new util.ArrayList[String]
        for (index <- 0 until finalOutputArrayList.size) {
          val arrayFromInputList = finalOutputArrayList.get(index)
          val rewardValue = finalProbabilityValueList.get(index)

          var tempString = arrayFromInputList.toString.replace('[', ' ').replace(']', ' ').replace(',', ':').replaceAll("\\s", "")
          var tempString2 = tempString
          tempString = tempString.replaceAll("-1", "2")
          val output = tempString + " " + String.valueOf(rewardValue)

          val indexInList1 = finalStringListForFile.indexOf(output)
          if (indexInList1 == -1) {
            finalStringListForFile.add(output)
          }
          tempString2 = tempString2.replaceAll("1", "2").replaceAll("-2", "1")
          val output2 = tempString2 + " " + String.valueOf(rewardValue)
          val indexInList2 = finalStringListForFile.indexOf(output2)

          if (indexInList2 == -1) {
            finalStringListForFile.add(output2)
          }
        }
        for (s <- finalStringListForFile.asScala) {
          writer.append(s)
          writer.append('\r')
          writer.append('\n')
          writer.flush()
        }
      } catch {
        case i: Exception => println(i.toString)
      } finally {
        if (writer != null) writer.close()
      }
    }
  }
}
