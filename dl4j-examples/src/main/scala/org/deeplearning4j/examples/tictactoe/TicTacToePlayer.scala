package org.deeplearning4j.examples.tictactoe

import java.io.{BufferedReader, FileReader, FileWriter}
import java.util
import java.util.concurrent.locks.{Lock, ReentrantLock}

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * <b>Developed by KIT Solutions Pvt. Ltd. (www.kitsol.com)</b> on 24-Aug-16.
  * This program does following tasks.
  * - loads tictactoe data file
  * - provide next best move depending on the previous passed
  * - reset the board when a game is over.
  * - checks whether game is finished.
  * - update probability of each move made in lost or won game when game is finished
  */
class TicTacToePlayer extends Runnable {

  // To synchronise access of stateList and stateProbabilityList.
  private[tictactoe] val lock: Lock = new ReentrantLock
  // holds path of data file to load data from.
  private var filePath: String = ""
  // holds data for state and probability loaded from data file.
  private val stateList: util.List[INDArray] = new util.ArrayList[INDArray]
  private val stateProbabilityList: util.List[Double] = new util.ArrayList[Double]

  /**
    * Stores a index position from stateList to hold all states from sateList list
    * e.g. if move made by first player is at the 5th position in stateList, then "indexListForPlayer1" will hold 5
    * This is required to update probability of particular state in stateList List when game is finished.
    * This is stored for both player separately for a single game and will be cleared at the end of the game after
    * updating probability.
    */
  private val indexListForPlayer1: util.List[Integer] = new util.ArrayList[Integer]
  private val indexListForPlayer2: util.List[Integer] = new util.ArrayList[Integer]
  // flag to control update of probability in a data file.
  private var updateAIAutomatic: Boolean = false
  //Stores game decision at any time. 0-For continue/Not started, 1-For Player1 wins,2-For Player2 wins,3-game Drawn
  private var gameDecision: Int = 0
  // controls whether data file is loaded or not. used in run() method.
  private var aiLoad: Boolean = false
  // class variable to hold number of games after which you want to update probability in data file.
  private var updateLimit: Int = 0
  // private class variable to control number of games played to allow program to update probability after updateLimit number of games.
  private var gameCounter: Int = 0
  // allows client class to set a flag whether update probability or not in data file. If this flag is false, updateLimit is of no use.
  private var updateAIFile: Boolean = false

  /**
    * Thread method to load or save data file asynchronously.
    */
  def run() {
    readStateAndRewardFromFile()
    while (true) {
      try {
        if (updateAIFile) {
          updateAIFile = false
          saveToFile()
        }
        Thread.sleep(100)
      } catch { case e: Exception =>
        println("Exception in File Updatable" + e.toString)
      }
    }
  }

  /**
    * to check whether data is loaded from data file into stateList and stateProbabilityList.
    */
  def isAILoad: Boolean = aiLoad

  /**
    * To retrieve best next move provided current board and player number (i.e. first or second player)
    */
  def getNextBestMove(board: INDArray, playerNumber: Int): INDArray = {
    var maxNumber: Double = 0
    var indexInArray: Int = 0
    var nextMove: INDArray = null
    val boardEmpty: Boolean = isBoardEmpty(board)
    if (!boardEmpty) {
      if (playerNumber == 1 && indexListForPlayer2.size == 0) {
        val indexInList: Int = stateList.indexOf(board)
        if (indexInList != -1) {
          indexListForPlayer2.add(indexInList)
        }
      } else if (playerNumber == 2 && indexListForPlayer1.size == 0) {
        val indexInList: Int = stateList.indexOf(board)
        if (indexInList != -1) {
          indexListForPlayer1.add(indexInList)
        }
      }
    }
    val listOfNextPossibleMove: util.List[INDArray] = getPossibleBoards(board, playerNumber)
    try {
      lock.lock()
      for (index <- 0 until listOfNextPossibleMove.size) {
        val positionArray: INDArray = listOfNextPossibleMove.get(index)
        val indexInStateList: Int = stateList.indexOf(positionArray)
        var probability: Double = 0
        if (indexInStateList != -1) {
          probability = stateProbabilityList.get(indexInStateList)
        }
        if (maxNumber <= probability) {
          maxNumber = probability
          indexInArray = indexInStateList
          nextMove = positionArray
        }
      }
    } catch {
      case e: Exception => println(e.toString)
    } finally {
      lock.unlock()
    }
    var isGameOver: Boolean = false
    if (playerNumber == 1) {
      indexListForPlayer1.add(indexInArray)
      isGameOver = isGameFinish(nextMove, isOdd = true)
    } else {
      indexListForPlayer2.add(indexInArray)
      isGameOver = isGameFinish(nextMove, isOdd = false)
    }
    if (isGameOver) {
      reset()
    }
    nextMove
  }

  /**
    * Checks if board is completely empty or not?
    */
  private def isBoardEmpty(board: INDArray): Boolean = {
    for (i <- 0 until board.length) {
      val digit: Double = board.getDouble(i)
      if (digit > 0) {
        false
      }
    }
    true
  }

  /**
    * resets index id list for all moves made by both users.
    */
  def reset() {
    indexListForPlayer1.clear()
    indexListForPlayer2.clear()
  }

  /**
    * Checks whether game is finished or not by checking three horizontal, three vertical and two diagonal moves made by any player.
    */
  private def isGameFinish(board: INDArray, isOdd: Boolean): Boolean = {
    var isGameOver: Boolean = false
    val boardPosition1: Double = board.getDouble(0)
    val boardPosition2: Double = board.getDouble(1)
    val boardPosition3: Double = board.getDouble(2)
    val boardPosition4: Double = board.getDouble(3)
    val boardPosition5: Double = board.getDouble(4)
    val boardPosition6: Double = board.getDouble(5)
    val boardPosition7: Double = board.getDouble(6)
    val boardPosition8: Double = board.getDouble(7)
    val boardPosition9: Double = board.getDouble(8)

    val position1: Boolean = if (isOdd) { board.getDouble(0) % 2.0 != 0 }
    else { board.getDouble(0) % 2.0 == 0 }
    val position2: Boolean = if (isOdd) { board.getDouble(1) % 2.0 != 0 }
    else { board.getDouble(1) % 2.0 == 0 }
    val position3: Boolean = if (isOdd) { board.getDouble(2) % 2.0 != 0 }
    else { board.getDouble(2) % 2.0 == 0 }
    val position4: Boolean = if (isOdd) { board.getDouble(3) % 2.0 != 0 }
    else { board.getDouble(3) % 2.0 == 0 }
    val position5: Boolean = if (isOdd) { board.getDouble(4) % 2.0 != 0 }
    else { board.getDouble(4) % 2.0 == 0 }
    val position6: Boolean = if (isOdd) { board.getDouble(5) % 2.0 != 0 }
    else { board.getDouble(5) % 2.0 == 0 }
    val position7: Boolean = if (isOdd) { board.getDouble(6) % 2.0 != 0 }
    else { board.getDouble(6) % 2.0 == 0 }
    val position8: Boolean = if (isOdd) { board.getDouble(7) % 2.0 != 0 }
    else { board.getDouble(7) % 2.0 == 0 }
    val position9: Boolean = if (isOdd) { board.getDouble(8) % 2.0 != 0 }
    else { board.getDouble(8) % 2.0 == 0 }
    if (((position1 && position2 && position3) && (boardPosition1 != 0 && boardPosition2 != 0 && boardPosition3 != 0)) ||
      ((position4 && position5 && position6) && (boardPosition4 != 0 && boardPosition5 != 0 && boardPosition6 != 0)) ||
      ((position7 && position8 && position9) && (boardPosition7 != 0 && boardPosition8 != 0 && boardPosition9 != 0)) ||
      ((position1 && position4 && position7) && (boardPosition1 != 0 && boardPosition4 != 0 && boardPosition7 != 0)) ||
      ((position2 && position5 && position8) && (boardPosition2 != 0 && boardPosition5 != 0 && boardPosition8 != 0)) ||
      ((position3 && position6 && position9) && (boardPosition3 != 0 && boardPosition6 != 0 && boardPosition9 != 0)) ||
      ((position1 && position5 && position9) && (boardPosition1 != 0 && boardPosition5 != 0 && boardPosition9 != 0)) ||
      ((position3 && position5 && position7) && (boardPosition3 != 0 && boardPosition5 != 0 && boardPosition7 != 0))) {

      gameCounter += 1

      if (isOdd) {
        gameDecision = 1
        updateReward(0, indexListForPlayer1) //Win player_1
        updateReward(1, indexListForPlayer2) //loose player_2
      } else {
        gameDecision = 2
        updateReward(0, indexListForPlayer2) //Win player_2
        updateReward(1, indexListForPlayer1) //loose player_1
      }
      isGameOver = true
      reset()
    } else {
      isGameOver = true

      var i: Int = 0
      while (i < 9 && board.getDouble(i).toInt != 0) { i += 1 }
      isGameOver = false
      gameDecision = 0

      //Draw for both player
      if (isGameOver) {
        gameDecision = 3
        gameCounter += 1
        updateReward(2, indexListForPlayer1)
        updateReward(2, indexListForPlayer2)
        reset()
      }
    }
    isGameOver
  }

  /**
    * Calculate probability of any won or lost or draw game at the end of the game and update stateList and stateProbabilityList.
    * It uses "Temporal Difference" formula to calculate probability of each game move.
    */
  private def updateReward(win: Int, playerMoveIndexList: util.List[Integer]) {
    if (!updateAIAutomatic) {
      return
    }

    if ((gameCounter >= updateLimit) && updateAIAutomatic) {
      gameCounter = 0
      updateAIFile = true
    }

    var probabilityValue: Double = 0.0
    var previousIndex: Int = 0
    try {
      lock.lock()
      for (p <- (0 until playerMoveIndexList.size).reverse) {
        previousIndex = playerMoveIndexList.get(p)
        if (p == (playerMoveIndexList.size - 1)) {
          if (win == 1) {
            probabilityValue = 0.0 //loose
          }
          else if (win == 0) {
            probabilityValue = 1.0 //Win
          }
          else {
            probabilityValue = 0.5 //Draw
          }
        }
        else {
          val probabilityFromPreviousStep: Double = stateProbabilityList.get(previousIndex)
          probabilityValue = probabilityFromPreviousStep + 0.1 * (probabilityValue - probabilityFromPreviousStep) //This is temporal difference formula for calculating reward for state
        }
        stateProbabilityList.set(previousIndex, probabilityValue.asInstanceOf[Double])
      }
    } catch {
      case e: Exception => {
        println(e.toString)
      }
    } finally {
      lock.unlock()
    }
  }

  /**
    * This function returns list of all possible boards states provided current board
    * This will be used to calculate best move for the next player to play
    */
  private def getPossibleBoards(board: INDArray, playerNumber: Int): util.List[INDArray] = {
    val returnList: util.List[INDArray] = new util.ArrayList[INDArray]
    for (i <- 0 until board.length) {
      val inputArray = Nd4j.zeros(1, 9)
      Nd4j.copy(board, inputArray)
      val digit: Double = board.getDouble(i)
      if (digit == 0) {
        inputArray.putScalar(Array[Int](0, i), playerNumber)
        returnList.add(inputArray)
      }
    }
    returnList
  }

  /**
    * This is the function to load data file into stateList and stateProbabilityList lists
    */
  private def readStateAndRewardFromFile() {
    try {
      val br: BufferedReader = new BufferedReader(new FileReader(filePath))
      try {
        var line: String = br.readLine
        lock.lock()
        while (line != null) {
          val input: INDArray = Nd4j.zeros(1, 9)
          val nextLine = line.split(" ")
          val tempLine1 = nextLine(0)
          val tempLine2 = nextLine(1)
          val testLine = tempLine1.split(":")
          for (i <- 0 until 9) {
            val number = testLine(i).toDouble
            input.putScalar(Array[Int](0, i), number)
          }
          val doubleNumber = tempLine2.toDouble
          stateList.add(input)
          stateProbabilityList.add(doubleNumber)
          aiLoad = true

          line = br.readLine
        }
      } catch {
        case e: Exception => println(e.toString)
      } finally {
        lock.unlock()
        if (br != null) br.close()
      }
    }
  }

  /**
    * Function to save current data in stateList and stateProbabilityList into data file.
    */
  private def saveToFile() {
    try {
      val writer: FileWriter = new FileWriter(filePath)
      try {
        lock.lock()
        for (index <- 0 until stateList.size) {
          val arrayFromInputList = stateList.get(index)
          val rewardValue = stateProbabilityList.get(index)
          val tempString = arrayFromInputList.toString.replace('[', ' ').replace(']', ' ').replace(',', ':').replaceAll("\\s", "")
          val output = tempString + " " + String.valueOf(rewardValue)
          writer.append(output)
          writer.append('\r')
          writer.append('\n')
          writer.flush()
        }
      } catch { case i: Exception =>
        println(i.toString)
      } finally {
        lock.unlock()
        if (writer != null) writer.close()
      }
    }
  }

  /**
    * returns current state of the game, i.e. won, lose, draw or in progress.
    */
  def getGameDecision: Int = {
    val currentResult: Int = gameDecision
    gameDecision = 0
    currentResult
  }

  /**
    * Sets a file name (with full path) to be used to load data from.
    */
  def setFilePath(filePath: String) {
    this.filePath = filePath
  }

  /**
    * This function is used to tell TicTacToePlayer to update probability in data file.
    * data file is not updated if you set this as false.
    * This property is false by default
    */
  def setAutoUpdate(updateAI: Boolean) {
    updateAIAutomatic = updateAI
  }

  /**
    * set a limit of number of games after which user wants to update data file from stateList and stateProbabilityList.
    */
  def setUpdateLimit(updateLimit: Int) {
    this.updateLimit = updateLimit
  }

  def addBoardToList(board: INDArray, playerNumber: Int) {
    val indexInStateList: Int = stateList.indexOf(board)
    if (indexInStateList != -1) {
      var isGameOver: Boolean = false
      if (playerNumber == 1) {
        indexListForPlayer1.add(indexInStateList)
        isGameOver = isGameFinish(board, isOdd = true)
      } else {
        indexListForPlayer2.add(indexInStateList)
        isGameOver = isGameFinish(board, isOdd = false)
      }
    }
  }
}
