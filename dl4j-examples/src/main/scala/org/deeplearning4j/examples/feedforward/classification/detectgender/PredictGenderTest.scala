package org.deeplearning4j.examples.feedforward.classification.detectgender

/**
  * Created by KITS on 9/14/2016.
  */

import java.awt.event.{ActionEvent, ActionListener}
import java.io.File
import javax.swing._

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

/**
  * "Linear" Data Classification Example
  *
  * Based on the data from Jason Baldridge:
  * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
  *
  * @author Josh Patterson
  * @author Alex Black (added plots)
  *
  */
object PredictGenderTest {
  @throws[Exception]
  def main(args: Array[String]) {
    val pgt: PredictGenderTest = new PredictGenderTest
    val t: Thread = new Thread(pgt)
    t.start()
    pgt.prepareInterface()
  }
}

class PredictGenderTest extends Runnable {
  private val row: Int = 0
  private var jd: JDialog = null
  private var jtf: JTextField = null
  private var jlbl: JLabel = null
  private var possibleCharacters: String = null
  private var gender: JLabel = null
  private var filePath: String = null
  private var btnNext: JButton = null
  private var genderLabel: JLabel = null
  private var model: MultiLayerNetwork = null

  def prepareInterface() {
    this.jd = new JDialog
    this.jd.getContentPane.setLayout(null)
    this.jd.setBounds(100, 100, 300, 250)
    this.jd.setLocationRelativeTo(null)
    this.jd.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE)
    this.jd.setTitle("Predict Gender By Name")
    //jd.add(jp);

    this.jlbl = new JLabel
    this.jlbl.setBounds(5, 10, 100, 20)
    this.jlbl.setText("Enter Name : ")
    this.jd.add(jlbl)

    this.jtf = new JTextField
    this.jtf.setBounds(105, 10, 150, 20)
    this.jd.add(jtf)

    this.genderLabel = new JLabel
    this.genderLabel.setBounds(5, 12, 70, 170)
    this.genderLabel.setText("Gender : ")
    this.jd.add(genderLabel)

    this.gender = new JLabel
    this.gender.setBounds(75, 12, 75, 170)
    this.jd.add(gender)

    this.btnNext = new JButton
    this.btnNext.setBounds(5, 150, 150, 20)
    this.btnNext.setText("Predict")

    this.btnNext.addActionListener(new ActionListener() {

      def actionPerformed(e: ActionEvent) {
        if (!jtf.getText.isEmpty) {
          val binaryData = getBinaryString(jtf.getText.toLowerCase)
          //System.out.println("binaryData : " + binaryData);
          val arr = binaryData.split(",")
          val db = new Array[Int](arr.length)
          val features = Nd4j.zeros(1, 235)
          for (i <- arr.indices) {
            features.putScalar(Array[Int](0, i), arr(i).toInt)
          }
          val predicted: INDArray = model.output(features)
          //System.out.println("output : " + predicted);
          if (predicted.getDouble(0) > predicted.getDouble(1))
            gender.setText("Female")
          else if (predicted.getDouble(0) < predicted.getDouble(1))
            gender.setText("Male")
          else
            gender.setText("Both male and female can have this name")
        } else
          gender.setText("Enter name please..")
      }
    })
    this.jd.add(this.btnNext)

    this.jd.setVisible(true)

  }

  private def getBinaryString(name: String): String = {
    var binaryString: String = ""
    for (j <- name.indices) {
      val fs = pad(Integer.toBinaryString(possibleCharacters.indexOf(name.charAt(j))), 5)
      binaryString = binaryString + fs
    }
    var diff = 0
    if (name.length < 47) {
      diff = 47 - name.length
      for (i <- 0 until diff) {
        binaryString = binaryString + "00000"
      }
    }
    var tempStr: String = ""
    for (i <- binaryString.indices) {
      tempStr = tempStr + binaryString.charAt(i) + ","
    }
    tempStr
  }

  private def pad(string: String, total_length: Int): String = {
    var str = string
    var diff = 0
    if (total_length > string.length) diff = total_length - string.length
    for (i <- 0 until diff) {
      str = "0" + str
    }
    str
  }

  def run() {
    try {
      this.filePath = System.getProperty("user.dir") + "/dl4j-examples/src/main/resources/PredictGender/Data/"
      this.possibleCharacters = " abcdefghijklmnopqrstuvwxyz"
      this.model = ModelSerializer.restoreMultiLayerNetwork(new File(this.filePath, "PredictGender.net"))
    } catch { case e: Exception =>
      println("Exception : " + e.getMessage)
    }
  }
}
