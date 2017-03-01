package org.deeplearning4j.examples.tictactoe

import java.awt._
import java.awt.event.{ActionEvent, ActionListener}
import java.io.File
import javax.swing._

import org.datavec.api.util.ClassPathResource
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * <b>Developed by KIT Solutions Pvt. Ltd.</b> (www.kitsol.com) on 24-Aug-16
  * This is a GUI to play game using trained network.
  */
object PlayTicTacToe {
  def main(args: Array[String]) {
    val game: PlayTicTacToe = new PlayTicTacToe
    game.renderGUI()
  }
}

/**
  * Constructor that loads trained data and initializes TicTacToePlayer object to play the game.
  * Also, initializes the GUI and display it.
  */
@throws[HeadlessException]
class PlayTicTacToe() extends JFrame {
  var filePath: String = ""
  try {
    filePath = new ClassPathResource("tictactoe").getFile.getAbsolutePath + File.separator + "AllMoveWithReward.txt"
  } catch {
    case e: Exception =>
      println("FilePathException" + e.toString)
  }


  private var playerInformation: String = "FirstPlayer:X"
  private val frame: JFrame = new JFrame("TicTacToe")
  private val gridMoveButton: Array[JButton] = new Array[JButton](9)
  private val startButton: JButton = new JButton("Start")
  private val switchButton: JButton = new JButton("Switch Player")
  private val infoLabel: JLabel = new JLabel(playerInformation)
  private var isAIFirstPlayer: Boolean = true
  private var xWon: Int = 0
  private var oWon: Int = 0
  private var draw: Int = 0
  private var ticTacToePlayer: TicTacToePlayer = new TicTacToePlayer
  private var aiLoad: Thread = new Thread(ticTacToePlayer)

  frame.setSize(350, 450)
  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.setVisible(true)
  frame.setResizable(false)

  ticTacToePlayer.setFilePath(filePath)
  aiLoad.start()

  ticTacToePlayer.setUpdateLimit(10)
  ticTacToePlayer.setAutoUpdate(true)

  /**
    * Create the GUI for tictactoe game with 9 move button,two utility button.
    */
  private def renderGUI() {
    val mainPanel = new JPanel(new BorderLayout)
    val menu = new JPanel(new BorderLayout)
    val tital = new JPanel(new BorderLayout)
    val game = new JPanel(new GridLayout(3, 3))

    frame.add(mainPanel)

    mainPanel.setPreferredSize(new Dimension(325, 425))
    menu.setPreferredSize(new Dimension(300, 50))
    tital.setPreferredSize(new Dimension(300, 50))
    game.setPreferredSize(new Dimension(300, 300))

    //Create the basic layout for game

    mainPanel.add(menu, BorderLayout.NORTH)
    mainPanel.add(tital, BorderLayout.AFTER_LINE_ENDS)
    mainPanel.add(game, BorderLayout.SOUTH)
    tital.add(infoLabel, BorderLayout.CENTER)
    menu.add(startButton, BorderLayout.WEST)
    menu.add(switchButton, BorderLayout.EAST)

    //Create the 9 Grid button on UI
    for (i <- 0 until 9) {
      gridMoveButton(i) = new JButton
      gridMoveButton(i).setText(" ")
      gridMoveButton(i).setVisible(true)
      gridMoveButton(i).setEnabled(false)
      gridMoveButton(i).addActionListener(new MyActionListener(i))
      game.add(gridMoveButton(i))
    }
    game.setVisible(true)
    startButton.setEnabled(false)
    switchButton.setEnabled(false)

    //Start Button Click Listener.
    startButton.addActionListener(new ActionListener() {
      def actionPerformed(e: ActionEvent) {
        reset() //Reset GUI
        changeGridButtonAccessibility(true)
        switchButton.setEnabled(true)
        if (isAIFirstPlayer) {
          val firstMove: INDArray = Nd4j.zeros(1, 9)
          val nextMove: INDArray = ticTacToePlayer.getNextBestMove(firstMove, 1)
          updateMoveOnBoard(nextMove)
        }
      }
    })

    //switch Button Click Listener
    switchButton.addActionListener(new ActionListener() {
      def actionPerformed(e: ActionEvent) {
        if (isAIFirstPlayer) {
          playerInformation = "FirstPlayer:O"
          isAIFirstPlayer = false
        } else {
          playerInformation = "FirstPlayer:X"
          isAIFirstPlayer = true
        }
        reset()
        updateInformation()
      }
    })

    while (!ticTacToePlayer.isAILoad) {
      try {
        Thread.sleep(10)
      } catch {
        case e: InterruptedException => {
          e.printStackTrace()
        }
      }
    }
    startButton.setEnabled(true)
    switchButton.setEnabled(true)
  }

  /**
    * Update the GUI depending upon the move provided by TicTacToePlayer or by manual player
    */
  private def updateMoveOnBoard(board: INDArray) {
    if (board == null) {
      return
    }
    if (isAIFirstPlayer) {
      for (i <- 0 until 9) {
        if (board.getDouble(i).toInt == 1) {
          gridMoveButton(i).setText("X")
        } else if (board.getDouble(i).toInt == 2) {
          gridMoveButton(i).setText("O")
        }
      }
    } else {
      for (i <- 0 until 9) {
        if (board.getDouble(i).toInt == 1) {
          gridMoveButton(i).setText("O")
        } else if (board.getDouble(i).toInt == 2) {
          gridMoveButton(i).setText("X")
        }
      }
    }
  }

  /**
    * Reset the button for and also reset the TicTacToe player object after game is finished.
    */
  private def reset() {
    for (i <- 0 until 9) {
      gridMoveButton(i).setText(" ")
      gridMoveButton(i).setEnabled(false)
    }
    ticTacToePlayer.reset()
  }

  /**
    * Enable or disable the game by disabling all buttons.
    */
  private def changeGridButtonAccessibility(enable: Boolean) {
    for (i <- 0 until 9) {
      gridMoveButton(i).setEnabled(enable)
    }
  }

  /**
    * This function gives the current state board in INDArray
    */
  private def getCurrentStateOfBoard: INDArray = {
    val positionArray: INDArray = Nd4j.zeros(1, 9)
    for (i <- 0 until 9) {
      val gridMoveButtonValue: String = gridMoveButton(i).getText
      if (isAIFirstPlayer) {
        if (gridMoveButtonValue == "X") {
          positionArray.putScalar(Array[Int](0, i), 1)
        } else if (gridMoveButtonValue == "O") {
          positionArray.putScalar(Array[Int](0, i), 2)
        }
      } else {
        if (gridMoveButtonValue == "O") {
          positionArray.putScalar(Array[Int](0, i), 1)
        } else if (gridMoveButtonValue == "X") {
          positionArray.putScalar(Array[Int](0, i), 2)
        }
      }
    }
    positionArray
  }

  /**
    * This method update the UI(Board) for click event on any button on the board and
    * immediately calls automatic user to play the next move.
    */
  private def userNextMove(indexPosition: Int) {
    val gridMoveButtonText: String = gridMoveButton(indexPosition).getText
    if (gridMoveButtonText == " ") {
      gridMoveButton(indexPosition).setText("O")
      playUsingAI()
    }
  }

  /**
    * This method is used for playing next move by machine itself.
    */
  private def playUsingAI() {
    val currentBoard: INDArray = getCurrentStateOfBoard
    var nextMove: INDArray = null
    var gameFinish: Boolean = false
    var gameState: Int = 0
    if (isAIFirstPlayer) {
      ticTacToePlayer.addBoardToList(currentBoard, 2)
      if (isGameFinish) {
        gameFinish = true
      } else {
        nextMove = ticTacToePlayer.getNextBestMove(currentBoard, 1)
        gameState = ticTacToePlayer.getGameDecision
      }
    } else {
      ticTacToePlayer.addBoardToList(currentBoard, 1)
      if (isGameFinish) {
        gameFinish = true
      } else {
        nextMove = ticTacToePlayer.getNextBestMove(currentBoard, 2)
        gameState = ticTacToePlayer.getGameDecision
      }
    }
    if (gameFinish) {
      updateInformation()
    } else {
      if (nextMove != null) {
        updateMoveOnBoard(nextMove)
      }
      if (gameState != 0) {
        if (isAIFirstPlayer) {
          if (gameState == 1) {
            xWon += 1
          } else if (gameState == 2) {
            oWon += 1
          } else {
            draw += 1
          }
        } else {
          if (gameState == 1) {
            oWon += 1
          } else if (gameState == 2) {
            xWon += 1
          } else {
            draw += 1
          }
        }
        updateInformation()
      }
    }
  }

  /**
    * Updates statisitical information about each user in terms of how many games both user won or lost and drawn also.
    */
  private def updateInformation() {
    val updateInformation: String = playerInformation + "    X:" + String.valueOf(xWon) + "    O:" + String.valueOf(oWon) + "    Draw:" + String.valueOf(draw)
    infoLabel.setText(updateInformation)
    changeGridButtonAccessibility(false)
  }

  /**
    * checks is game is finished. It checks it from TicTacToePlayer object.
    */
  private def isGameFinish: Boolean = {
    val result: Int = ticTacToePlayer.getGameDecision
    if (result != 0) {
      if (result == 1) {
        if (isAIFirstPlayer) {
          xWon += 1
        } else {
          oWon += 1
        }
      }
      else if (result == 2) {
        if (!isAIFirstPlayer) {
          xWon += 1
        } else {
          oWon += 1
        }
      } else {
        draw += 1
      }
      true
    }
    false
  }

  /**
    * This is Action listener for move buttons
    */
  class MyActionListener(var index: Int) extends ActionListener {
    def actionPerformed(e: ActionEvent) {
      userNextMove(index)
    }
  }

}
