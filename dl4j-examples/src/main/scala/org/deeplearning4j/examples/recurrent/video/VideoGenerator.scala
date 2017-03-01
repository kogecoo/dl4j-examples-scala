package org.deeplearning4j.examples.recurrent.video

import java.awt._
import java.awt.geom.{Arc2D, Ellipse2D, Line2D, Rectangle2D}
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util.Random

import org.apache.commons.io.FilenameUtils
import org.jcodec.api.SequenceEncoder

/** A support class for generating a synthetic video data set
  * Nothing here is specific to DL4J
  *
  * @author Alex Black
  */
object VideoGenerator {
  val NUM_SHAPES = 4 //0=circle, 1=square, 2=arc, 3=line
  val MAX_VELOCITY = 3
  val SHAPE_SIZE = 25
  val SHAPE_MIN_DIST_FROM_EDGE = 15
  val DISTRACTOR_MIN_DIST_FROM_EDGE = 0
  val LINE_STROKE_WIDTH = 6 //Width of line (line shape only)
  val lineStroke = new BasicStroke(LINE_STROKE_WIDTH)
  val MIN_FRAMES = 10 //Minimum number of frames the target shape to be present
  val MAX_NOISE_VALUE = 0.5f

  @throws[Exception]
  private def generateVideo(path: String, nFrames: Int, width: Int, height: Int, numShapes: Int, r: Random, backgroundNoise: Boolean, numDistractorsPerFrame: Int): Array[Int] = {
    //First: decide where transitions between one shape and another are
    val temp = (0 until numShapes).map({ _ => r.nextDouble() })
    val sum = temp.sum
    val rns = temp.map { _  / sum }

    val startFrames: Array[Int] = new Array[Int](numShapes)
    startFrames(0) = 0

    (1 until numShapes).map { i =>
      (startFrames(i - 1) + MIN_FRAMES + rns(i) * (nFrames - numShapes * MIN_FRAMES)).toInt
    }

    //Randomly generate shape positions, velocities, colors, and type
    val shapeTypes = new Array[Int](numShapes)
    val initialX = new Array[Int](numShapes)
    val initialY = new Array[Int](numShapes)
    val velocityX = new Array[Double](numShapes)
    val velocityY = new Array[Double](numShapes)
    val color = new Array[Color](numShapes)

    for (i <- 0 until numShapes) {
      shapeTypes(i) = r.nextInt(NUM_SHAPES)
      initialX(i) = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE)
      initialY(i) = SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE - 2 * SHAPE_MIN_DIST_FROM_EDGE)
      velocityX(i) = -1 + 2 * r.nextDouble
      velocityY(i) = -1 + 2 * r.nextDouble
      color(i) = new Color(r.nextFloat, r.nextFloat, r.nextFloat)
    }

    //Generate a sequence of BufferedImages with the given shapes, and write them to the video
    val enc = new SequenceEncoder(new File(path))
    var currShape = 0
    val labels = new Array[Int](nFrames)

    for (i <- 0 until nFrames) {
      if (currShape < numShapes - 1 && i >= startFrames(currShape + 1)) {
        currShape += 1; currShape - 1
      }
      val bi: BufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      val g2d: Graphics2D = bi.createGraphics
      g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
      g2d.setBackground(Color.BLACK)
      if (backgroundNoise) {
        for (x <- 0 until width) {
          for (y <- 0 until height) {
            bi.setRGB(x, y, new Color(r.nextFloat * MAX_NOISE_VALUE, r.nextFloat * MAX_NOISE_VALUE, r.nextFloat * MAX_NOISE_VALUE).getRGB)
          }
        }
      }
      g2d.setColor(color(currShape))
      //Position of shape this frame
      val currX = (initialX(currShape) + (i - startFrames(currShape)) * velocityX(currShape) * MAX_VELOCITY).toInt
      val currY = (initialY(currShape) + (i - startFrames(currShape)) * velocityY(currShape) * MAX_VELOCITY).toInt
      //Render the shape
      shapeTypes(currShape) match {
        case 0 =>  //Circle
          g2d.fill(new Ellipse2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
        case 1 =>  //Square
          g2d.fill(new Rectangle2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
        case 2 =>  //Arc
          g2d.fill(new Arc2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE))
        case 3 => //Line
          g2d.setStroke(lineStroke)
          g2d.draw(new Line2D.Double(currX, currY, currX + SHAPE_SIZE, currY + SHAPE_SIZE))
        case _ =>
          throw new RuntimeException
      }

      //Add some distractor shapes, which are present for one frame only
      for (j <- 0 until numDistractorsPerFrame) {
        val distractorShapeIdx: Int = r.nextInt(NUM_SHAPES)
        val distractorX: Int = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE)
        val distractorY: Int = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE)
        g2d.setColor(new Color(r.nextFloat, r.nextFloat, r.nextFloat))
        distractorShapeIdx match {
          case 0 =>
            g2d.fill(new Ellipse2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
          case 1 =>
            g2d.fill(new Rectangle2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
          case 2 =>
            g2d.fill(new Arc2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE))
          case 3 =>
            g2d.setStroke(lineStroke)
            g2d.draw(new Line2D.Double(distractorX, distractorY, distractorX + SHAPE_SIZE, distractorY + SHAPE_SIZE))
          case _ =>
            throw new RuntimeException
        }
      }
      enc.encodeImage(bi)
      g2d.dispose()
      labels(i) = shapeTypes(currShape)
    }
    enc.finish() //write .mp4
    labels
  }

  @throws[Exception]
  def generateVideoData(outputFolder: String, filePrefix: String, nVideos: Int, nFrames: Int, width: Int, height: Int, numShapesPerVideo: Int, backgroundNoise: Boolean, numDistractorsPerFrame: Int, seed: Long) {
    val r = new Random(seed)
    for (i <- 0 until nVideos) {
      val videoPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".mp4")
      val labelsPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".txt")
      val labels  = generateVideo(videoPath, nFrames, width, height, numShapesPerVideo, r, backgroundNoise, numDistractorsPerFrame)

      //Write labels to text file
      val sb = new StringBuilder
      for (j <- labels.indices) {
        sb.append(labels(j))
        if (j != labels.length - 1) sb.append("\n")
      }
      Files.write(Paths.get(labelsPath), sb.toString.getBytes("utf-8"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
    }
  }
}
