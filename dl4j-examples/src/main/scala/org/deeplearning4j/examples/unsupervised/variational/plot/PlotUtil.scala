package org.deeplearning4j.examples.unsupervised.variational.plot

import java.awt._
import java.awt.image.BufferedImage
import java.util
import javax.swing._
import javax.swing.event.{ChangeEvent, ChangeListener}

import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.{PlotOrientation, XYPlot}
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.jfree.chart.{ChartFactory, ChartPanel, JFreeChart}
import org.jfree.data.xy.{XYDataset, XYSeries, XYSeriesCollection}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._

/**
  * Plotting methods for the VariationalAutoEncoder example
  *
  * @author Alex Black
  */
object PlotUtil {

  def plotData(xyVsIter: util.List[INDArray], labels: INDArray, axisMin: Double, axisMax: Double, plotFrequency: Int) {

    val panel = new ChartPanel(createChart(xyVsIter.get(0), labels, axisMin, axisMax))
    val slider = new JSlider(0, xyVsIter.size - 1, 0)
    slider.setSnapToTicks(true)

    val f = new JFrame
    slider.addChangeListener(new ChangeListener() {
      private var lastPanel = panel

      def stateChanged(e: ChangeEvent) {
        val slider = e.getSource.asInstanceOf[JSlider]
        val value = slider.getValue
        val panel = new ChartPanel(createChart(xyVsIter.get(value), labels, axisMin, axisMax))
        if (lastPanel != null) {
          f.remove(lastPanel)
        }
        lastPanel = panel
        f.add(panel, BorderLayout.CENTER)
        f.setTitle(getTitle(value, plotFrequency))
        f.revalidate()
      }
    })

    f.setLayout(new BorderLayout)
    f.add(slider, BorderLayout.NORTH)
    f.add(panel, BorderLayout.CENTER)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle(getTitle(0, plotFrequency))

    f.setVisible(true)
  }

  private def getTitle(recordNumber: Int, plotFrequency: Int): String = {
    "MNIST Test Set - Latent Space Encoding at Training Iteration " + recordNumber * plotFrequency
  }

  //Test data
  private def createDataSet(features: INDArray, labelsOneHot: INDArray): XYDataset = {
    val nRows = features.rows
    val nClasses = labelsOneHot.columns
    val series  = new Array[XYSeries](nClasses)
    for (i <- 0 until nClasses) {
      series(i) = new XYSeries(String.valueOf(i))
    }
    val classIdx = Nd4j.argMax(labelsOneHot, 1)
    for (i <- 0 until nRows) {
      val idx = classIdx.getInt(i)
      series(idx).add(features.getDouble(i, 0), features.getDouble(i, 1))
    }
    val c = new XYSeriesCollection
    for (s <- series) c.addSeries(s)
    c
  }

  private def createChart(features: INDArray, labels: INDArray, axisMin: Double, axisMax: Double): JFreeChart = {

    val dataset = createDataSet(features, labels)

    val chart = ChartFactory.createScatterPlot("Variational Autoencoder Latent Space - MNIST Test Set",
      "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false)

    val plot = chart.getPlot.asInstanceOf[XYPlot]
    plot.getRenderer.setBaseOutlineStroke(new BasicStroke(0))
    plot.setNoDataMessage("NO DATA")

    plot.setDomainPannable(false)
    plot.setRangePannable(false)
    plot.setDomainZeroBaselineVisible(true)
    plot.setRangeZeroBaselineVisible(true)

    plot.setDomainGridlineStroke(new BasicStroke(0.0f))
    plot.setDomainMinorGridlineStroke(new BasicStroke(0.0f))
    plot.setDomainGridlinePaint(Color.blue)
    plot.setRangeGridlineStroke(new BasicStroke(0.0f))
    plot.setRangeMinorGridlineStroke(new BasicStroke(0.0f))
    plot.setRangeGridlinePaint(Color.blue)

    plot.setDomainMinorGridlinesVisible(true)
    plot.setRangeMinorGridlinesVisible(true)

    val renderer: XYLineAndShapeRenderer = plot.getRenderer.asInstanceOf[XYLineAndShapeRenderer]
    renderer.setSeriesOutlinePaint(0, Color.black)
    renderer.setUseOutlinePaint(true)
    val domainAxis: NumberAxis = plot.getDomainAxis.asInstanceOf[NumberAxis]
    domainAxis.setAutoRangeIncludesZero(false)
    domainAxis.setRange(axisMin, axisMax)

    domainAxis.setTickMarkInsideLength(2.0f)
    domainAxis.setTickMarkOutsideLength(2.0f)

    domainAxis.setMinorTickCount(2)
    domainAxis.setMinorTickMarksVisible(true)

    val rangeAxis: NumberAxis = plot.getRangeAxis.asInstanceOf[NumberAxis]
    rangeAxis.setTickMarkInsideLength(2.0f)
    rangeAxis.setTickMarkOutsideLength(2.0f)
    rangeAxis.setMinorTickCount(2)
    rangeAxis.setMinorTickMarksVisible(true)
    rangeAxis.setRange(axisMin, axisMax)
    chart
  }

  class MNISTLatentSpaceVisualizer(var imageScale: Double, var digits: util.List[INDArray] //Digits (as row vectors), one per INDArray
                                   , var plotFrequency: Int) {
    private val gridWidth: Int = Math.sqrt(digits.get(0).size(0)).toInt //Assume square, nxn rows

    private def getTitle(recordNumber: Int): String = {
      "Reconstructions Over Latent Space at Training Iteration " + recordNumber * plotFrequency
    }

    def visualize() {
      val frame = new JFrame
      frame.setTitle(getTitle(0))
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
      frame.setLayout(new BorderLayout)

      val panel = new JPanel
      panel.setLayout(new GridLayout(0, gridWidth))

      val slider = new JSlider(0, digits.size - 1, 0)
      slider.addChangeListener(new ChangeListener() {
        def stateChanged(e: ChangeEvent) {
          val slider: JSlider = e.getSource.asInstanceOf[JSlider]
          val value: Int = slider.getValue
          panel.removeAll()
          val list: util.List[JLabel] = getComponents(value)
          for (image <- list.asScala) {
            panel.add(image)
          }
          frame.setTitle(getTitle(value))
          frame.revalidate()
        }
      })
      frame.add(slider, BorderLayout.NORTH)

      val list = getComponents(0)
      for (image <- list.asScala) {
        panel.add(image)
      }

      frame.add(panel, BorderLayout.CENTER)
      frame.setVisible(true)
      frame.pack()
    }

    private def getComponents(idx: Int): util.List[JLabel] = {
      val images: util.List[JLabel] = new util.ArrayList[JLabel]
      val temp: util.List[INDArray] = new util.ArrayList[INDArray]
      for (i <- 0 until digits.get(idx).size(0)) {
        temp.add(digits.get(idx).getRow(i))
      }
      for (arr <- temp.asScala) {
        val bi: BufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
        for (i <- 0 until 784) {
          bi.getRaster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i)).toInt)
        }
        val orig: ImageIcon = new ImageIcon(bi)
        val imageScaled: Image = orig.getImage.getScaledInstance((imageScale * 28).toInt, (imageScale * 28).toInt, Image.SCALE_REPLICATE)
        val scaled: ImageIcon = new ImageIcon(imageScaled)
        images.add(new JLabel(scaled))
      }
      images
    }
  }

}
