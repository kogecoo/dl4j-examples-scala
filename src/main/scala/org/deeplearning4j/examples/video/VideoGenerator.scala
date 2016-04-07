package org.deeplearning4j.examples.video

import org.apache.commons.io.FilenameUtils
import org.jcodec.api.SequenceEncoder

import java.awt._
import java.awt.geom.Arc2D
import java.awt.geom.Ellipse2D
import java.awt.geom.Line2D
import java.awt.geom.Rectangle2D
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardOpenOption
import java.util.Random

import scala.collection.mutable

/**A support class for generating a synthetic video data set
 * Nothing here is specific to DL4J
  *
  * @author Alex Black
 */
object VideoGenerator {

    final val NUM_SHAPES = 4;  //0=circle, 1=square, 2=arc, 3=line
    final val MAX_VELOCITY = 3
    final val SHAPE_SIZE = 25
    final val SHAPE_MIN_DIST_FROM_EDGE = 15
    final val DISTRACTOR_MIN_DIST_FROM_EDGE = 0
    final val LINE_STROKE_WIDTH = 6;  //Width of line (line shape only)
    final val lineStroke = new BasicStroke(LINE_STROKE_WIDTH)
    final val MIN_FRAMES = 10;    //Minimum number of frames the target shape to be present
    final val MAX_NOISE_VALUE = 0.5f

    private[this] def generateVideo(path: String, nFrames: Int, width: Int, height: Int, numShapes: Int, r: Random,
                                      backgroundNoise: Boolean, numDistractorsPerFrame: Int): Array[Int] = {

        //First: decide where transitions between one shape and another are
        val rands = (0 until numShapes).map { i => r.nextDouble() }
        val sum = rands.sum
        val normalized = rands.map { _ / sum }

        val startFrames: Seq[Int] = (1 until numShapes).scanLeft(0) { case (acc, i) =>
            (acc + MIN_FRAMES + normalized(i) * (nFrames - numShapes * MIN_FRAMES)).toInt
        }

        //Randomly generate shape positions, velocities, colors, and type
        val shapeTypes = Seq.fill(numShapes)(r.nextInt(NUM_SHAPES))
        val initialX   = Seq.fill(numShapes)(SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE - 2*SHAPE_MIN_DIST_FROM_EDGE ))
        val initialY   = Seq.fill(numShapes)(SHAPE_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE - 2*SHAPE_MIN_DIST_FROM_EDGE))
        val velocityX  = Seq.fill(numShapes)(-1 + 2 * r.nextDouble())
        val velocityY  = Seq.fill(numShapes)(-1 + 2 * r.nextDouble())
        val color      = Seq.fill(numShapes)(new Color(r.nextFloat(), r.nextFloat(), r.nextFloat()))

        //Generate a sequence of BufferedImages with the given shapes, and write them to the video
        val enc = new SequenceEncoder(new File(path))
        var currShape = 0
        val labelsBuilder = mutable.ArrayBuilder.make[Int]

        (0 until nFrames).foreach { i =>
            if (currShape < numShapes - 1 && i >= startFrames(currShape + 1)) currShape += 1

            val bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
            val g2d: Graphics2D = bi.createGraphics()
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON)
            g2d.setBackground(Color.BLACK)

            if(backgroundNoise){
                (0 until width).foreach { x =>
                    (0 until height).foreach { y =>
                        bi.setRGB(x,y,new Color(r.nextFloat()*MAX_NOISE_VALUE,r.nextFloat()*MAX_NOISE_VALUE,r.nextFloat()*MAX_NOISE_VALUE).getRGB)
                    }
                }
            }

            g2d.setColor(color(currShape))

            //Position of shape this frame
            val currX = (initialX(currShape) + (i - startFrames(currShape)) * velocityX(currShape) * MAX_VELOCITY).toInt
            val currY = (initialY(currShape) + (i - startFrames(currShape)) * velocityY(currShape) * MAX_VELOCITY).toInt

            //Render the shape
            shapeTypes(currShape) match {
                case 0 => {
                    //Circle
                    g2d.fill(new Ellipse2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
                }
                case 1 => {
                    //Square
                    g2d.fill(new Rectangle2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE))
                }
                case 2 => {
                    //Arc
                    g2d.fill(new Arc2D.Double(currX, currY, SHAPE_SIZE, SHAPE_SIZE, 315, 225, Arc2D.PIE))
                }
                case 3 => {
                    //Line
                    g2d.setStroke(lineStroke)
                    g2d.draw(new Line2D.Double(currX, currY, currX + SHAPE_SIZE, currY + SHAPE_SIZE))
                }
                case _ => throw new RuntimeException()
            }

            //Add some distractor shapes, which are present for one frame only
            (0 until numDistractorsPerFrame).foreach { _ =>
                val distractorShapeIdx = r.nextInt(NUM_SHAPES)

                val distractorX = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(width - SHAPE_SIZE)
                val distractorY = DISTRACTOR_MIN_DIST_FROM_EDGE + r.nextInt(height - SHAPE_SIZE)

                g2d.setColor(new Color(r.nextFloat(), r.nextFloat(), r.nextFloat()))

                distractorShapeIdx match {
                    case 0 => g2d.fill(new Ellipse2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
                    case 1 => g2d.fill(new Rectangle2D.Double(distractorX, distractorY, SHAPE_SIZE, SHAPE_SIZE))
                    case 2 => g2d.fill(new Arc2D.Double(distractorX,distractorY,SHAPE_SIZE,SHAPE_SIZE,315,225,Arc2D.PIE))
                    case 3 => {
                        g2d.setStroke(lineStroke)
                        g2d.draw(new Line2D.Double(distractorX, distractorY, distractorX + SHAPE_SIZE, distractorY + SHAPE_SIZE))
                    }
                    case _ => throw new RuntimeException()
                }
            }

            enc.encodeImage(bi)
            g2d.dispose()
            labelsBuilder += shapeTypes(currShape)
        }
        enc.finish();   //write .mp4

        labelsBuilder.result()
    }

    def generateVideoData(outputFolder: String, filePrefix: String, nVideos: Int, nFrames: Int,
                                         width: Int, height: Int, numShapesPerVideo: Int, backgroundNoise: Boolean,
                                         numDistractorsPerFrame: Int, seed: Long) = {
        val r = new Random(seed)

        (0 until nVideos).foreach { i =>
            val videoPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".mp4")
            val labelsPath = FilenameUtils.concat(outputFolder, filePrefix + "_" + i + ".txt")
            val labels = generateVideo(videoPath, nFrames, width, height, numShapesPerVideo, r, backgroundNoise, numDistractorsPerFrame)

            //Write labels to text file
            val sb = new StringBuilder()
            labels.indices.foreach { j =>
                sb.append(labels(j))
                if (j != labels.length - 1) sb.append("\n")
            }
            Files.write(Paths.get(labelsPath), sb.toString().getBytes("utf-8"), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
        }
    }
}
