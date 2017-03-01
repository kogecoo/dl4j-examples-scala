package org.deeplearning4j.examples.feedforward.classification.detectgender

/**
  * Created by KIT Solutions (www.kitsol.com) on 11/7/2016.
  */

import java.util
import java.io.{File, IOException}
import java.nio.charset.Charset
import java.nio.file.{Files, Paths}

import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.impl.LineRecordReader
import org.datavec.api.split.{FileSplit, InputSplit, InputStreamInputSplit}
import org.datavec.api.writable.{DoubleWritable, Writable}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

/**
  * GenderRecordReader class does following job
  * - Initialize method reads .CSV file as specified in Labels in constructor
  * - It loads person name and gender data into binary converted data
  * - creates binary string iterator which can be used by RecordReaderDataSetIterator
  *
  * Constructor to allow client application to pass List of possible Labels
  *
  * @param labels - List of String that client application pass all possible labels, in our case "M" and "F"
  */
class GenderRecordReader(labels: List[String]) extends LineRecordReader {

  // Final list that contains actual binary data generated from person name, it also contains label (1 or 0) at the end
  private var names = new mutable.ArrayBuffer[String]
  // String to hold all possible alphabets from all person names in raw data
  // This String is used to convert person name to binary string seperated by comma
  private var possibleCharacters = ""
  // holds length of largest name out of all person names
  var maxLengthName = 0
  // holds total number of names including both male and female names.
  // This variable is not used in PredictGenderTrain.java
  var totalRecords = 0
  // iterator for List "names" to be used in next() method
  private var iter: Iterator[String] = null

  /**
    * returns total number of records in List "names"
    *
    * @return - totalRecords
    */

  /**
    * This function does following steps
    * - Looks for the files with the name (in specified folder) as specified in labels set in constructor
    * - File must have person name and gender of the person (M or F),
    *   e.g. Deepan,M
    * Trupesh,M
    * Vinay,M
    * Ghanshyam,M
    *
    * Meera,F
    * Jignasha,F
    * Chaku,F
    *
    * - File for male and female names must be different, like M.csv, F.csv etc.
    * - populates all names in temporary list
    * - generate binary string for each alphabet for all person names
    * - combine binary string for all alphabets for each name
    * - find all unique alphabets to generate binary string mentioned in above step
    * - take equal number of records from all files. To do that, finds minimum record from all files, and then takes
    * that number of records from all files to keep balance between data of different labels.
    * - Note : this function uses stream() feature of Java 8, which makes processing faster. Standard method to process file takes more than 5-7 minutes.
    * using stream() takes approximately 800-900 ms only.
    * - Final converted binary data is stored List<String> names variable
    * - sets iterator from "names" list to be used in next() function
    *
    * @param split - user can pass directory containing .CSV file for that contains names of male or female
    * @throws IOException
    * @throws InterruptedException
    */
  @throws[IOException]
  @throws[InterruptedException]
  override def initialize(split: InputSplit) {
    if (split.isInstanceOf[FileSplit]) {

      val locations = split.locations
      if (locations != null && locations.length > 1) {
        var longestName = ""
        var uniqueCharactersTempString = ""
        val tempNames = mutable.ArrayBuffer.empty[(String, Array[String])]
        for (location <- locations) {
          val file = new File(location)
          val temp = labels.filter { line =>
            file.getName.equals(line + ".csv")
          }
          if (temp.nonEmpty) {
            val path = Paths.get(file.getAbsolutePath)
            val tempList = Files.readAllLines(path, Charset.defaultCharset).asScala
              .map({ element => element.split(',').head})

            val optional = if (tempList.nonEmpty) {
              val n = tempList.foldLeft("") { case (a, b) => if (a.length >= b.length) a else b }
              Some(n)
            } else None

            if (optional.nonEmpty && optional.get.length > longestName.length) longestName = optional.get

            uniqueCharactersTempString = uniqueCharactersTempString + tempList.toString
            tempNames += ((temp.head, tempList.toArray))
          } // else throw new InterruptedException("File missing for any of the specified labels")
        }

        println(s"${longestName.length} --- ")
        this.maxLengthName = longestName.length

        val chars = uniqueCharactersTempString.toSet[Char].toArray.sorted
        val unique = uniqueCharactersTempString.toSet[Char].toSeq.mkString
          .replaceAll("\\[", "").replaceAll("\\]", "").replaceAll(",", "") match {
          case u if u.startsWith(" ") => " " + u.trim()
          case u => u
        }

        this.possibleCharacters = unique
        val tempPair = tempNames.head
        var minSize = tempPair._2.length
        for (i <- 1 until tempNames.size) {
            if (minSize > tempNames(i)._2.length) minSize = tempNames(i)._2.length
        }
        val oneMoreTempNames = mutable.ArrayBuffer.empty[(String, Array[String])]
        var i: Int = 0
        for (i <- tempNames.indices) {
          val diff = Math.abs(minSize - tempNames(i)._2.length)
          val tempList = if (tempNames(i)._2.length > minSize) {
            val t = tempNames(i)._2
            t.slice(0, t.length - diff + 1)
          } else tempNames(i)._2
          val tempNewPair = (tempNames(i)._1, tempList)
          oneMoreTempNames += tempNewPair
        }
        tempNames.clear()
        val secondMoreTempNames = mutable.ArrayBuffer.empty[(String, Array[String])]
        for (i <- oneMoreTempNames.indices) {
          val gender = if (oneMoreTempNames(i)._1 == "M") 1 else 0
          val secondList = oneMoreTempNames(i)._2.map({element =>
            getBinaryString(element.split(',').head, gender)
          })
          val secondTempPair = ((oneMoreTempNames(i)._1, secondList))
          secondMoreTempNames += secondTempPair
        }
        oneMoreTempNames.clear()

        for (i <- secondMoreTempNames.indices) {
          names ++= secondMoreTempNames(i)._2
        }
        secondMoreTempNames.clear()
        this.totalRecords = names.size
        names = Random.shuffle(names)
        this.iter = names.iterator
      } else throw new InterruptedException("File missing for any of the specified labels")
    } else if (split.isInstanceOf[InputStreamInputSplit]) {
      println("InputStream Split found...Currently not supported")
      throw new InterruptedException("File missing for any of the specified labels")
    }
  }

  /**
    * - takes onme record at a time from names list using iter iterator
    * - stores it into Writable List and returns it
    *
    * @return
    */
  override def next(): util.List[Writable] = {
    if (iter.hasNext) {
      val ret = mutable.ArrayBuffer.empty[Writable]
      val currentRecord = iter.next
      val temp = currentRecord.split(",")
      for (i <- temp.indices) {
        ret += new DoubleWritable(temp(i).toDouble)
      }
      ret.toList.asJava
    } else throw new IllegalStateException("no more elements")
  }

  override def hasNext: Boolean = {
    if (iter != null) {
      iter.hasNext
    } else {
      throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist")
    }
  }

  @throws[IOException]
  override def close() { }

  override def setConf(conf: Configuration) {
    this.conf = conf
  }

  override def getConf: Configuration = {
    conf
  }

  override def reset() {
    this.iter = names.iterator
  }

  /**
    * This function gives binary string for full name string
    * - It uses "PossibleCharacters" string to find the decimal equivalent to any alphabet from it
    * - generate binary string for each alphabet
    * - left pads binary string for each alphabet to make it of size 5
    * - combine binary string for all alphabets of a name
    * - Right pads complete binary string to make it of size that is the size of largest name to keep all name length of equal size
    * - appends label value (1 or 0 for male or female respectively)
    *
    * @param name   - person name to be converted to binary string
    * @param gender - variable to decide value of label to be added to name's binary string at the end of the string
    * @return
    */
  private def getBinaryString(name: String, gender: Int): String = {
    var binaryString: String = ""
    for (j <- name.indices) {
      val fs = org.apache.commons.lang3.StringUtils.leftPad(Integer.toBinaryString(this.possibleCharacters.indexOf(name.charAt(j))), 5, "0")
      binaryString = binaryString + fs
    }
    //binaryString = String.format("%-" + this.maxLengthName*5 + "s",binaryString).replace(' ','0'); // this takes more time than StringUtils, hence commented
    binaryString = org.apache.commons.lang3.StringUtils.rightPad(binaryString, this.maxLengthName * 5, "0")
    binaryString = binaryString.replaceAll(".(?!$)", "$0,")
    //println("binary String : " + binaryString);
    binaryString + "," + String.valueOf(gender)
  }
}
