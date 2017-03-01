package org.deeplearning4j.examples.feedforward.classification

import org.jfree.chart.{ChartPanel, ChartUtilities, JFreeChart}
import org.jfree.chart.axis.{AxisLocation, NumberAxis}
import org.jfree.chart.block.BlockBorder
import org.jfree.chart.plot.{DatasetRenderingOrder, XYPlot}
import org.jfree.chart.renderer.GrayPaintScale
import org.jfree.chart.renderer.xy.{XYBlockRenderer, XYLineAndShapeRenderer}
import org.jfree.chart.title.PaintScaleLegend
import org.jfree.data.xy._
import org.jfree.ui.{RectangleEdge, RectangleInsets}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.factory.Nd4j

import java.awt._
import javax.swing._

/** Simple plotting methods for the MLPClassifier examples
  *
  * @author Alex Black
  */
object PlotUtil {

  /** Plot the training data. Assume 2d input, classification output
    *
    * @param features      Training data features
    * @param labels        Training data labels (one-hot representation)
    * @param backgroundIn  sets of x,y points in input space, plotted in the background
    * @param backgroundOut results of network evaluation at points in x,y points in space
    * @param nDivisions    Number of points (per axis, for the backgroundIn/backgroundOut arrays)
    */
  def plotTrainingData(features: INDArray, labels: INDArray, backgroundIn: INDArray, backgroundOut: INDArray, nDivisions: Int) {
    val mins: Array[Double] = backgroundIn.min(0).data.asDouble
    val maxs: Array[Double] = backgroundIn.max(0).data.asDouble

    val backgroundData = createBackgroundData(backgroundIn, backgroundOut)
    val panel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTrain(features, labels)))

    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Training Data")

    f.setVisible(true)
  }

  /** Plot the training data. Assume 2d input, classification output
    *
    * @param features      Training data features
    * @param labels        Training data labels (one-hot representation)
    * @param predicted     Network predictions, for the test points
    * @param backgroundIn  sets of x,y points in input space, plotted in the background
    * @param backgroundOut results of network evaluation at points in x,y points in space
    * @param nDivisions    Number of points (per axis, for the backgroundIn/backgroundOut arrays)
    */
  def plotTestData(features: INDArray, labels: INDArray, predicted: INDArray, backgroundIn: INDArray, backgroundOut: INDArray, nDivisions: Int) {

    val mins = backgroundIn.min(0).data.asDouble
    val maxs = backgroundIn.max(0).data.asDouble

    val backgroundData = createBackgroundData(backgroundIn, backgroundOut)
    val panel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTest(features, labels, predicted)))

    val f = new JFrame
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()
    f.setTitle("Test Data")

    f.setVisible(true)
  }

  /** Create data for the background data set
    */
  private def createBackgroundData(backgroundIn: INDArray, backgroundOut: INDArray): XYZDataset = {
    val nRows: Int = backgroundIn.rows
    val xValues: Array[Double] = new Array[Double](nRows)
    val yValues: Array[Double] = new Array[Double](nRows)
    val zValues: Array[Double] = new Array[Double](nRows)
    for (i <- 0 until nRows) {
      xValues(i) = backgroundIn.getDouble(i, 0)
      yValues(i) = backgroundIn.getDouble(i, 1)
      zValues(i) = backgroundOut.getDouble(i)
    }
    val dataset = new DefaultXYZDataset
    dataset.addSeries("Series 1", Array[Array[Double]](xValues, yValues, zValues))
    dataset
  }

  //Training data
  private def createDataSetTrain(features: INDArray, labels: INDArray): XYDataset = {
    val nRows = features.rows

    val nClasses = labels.columns

    val series = (for (i <- 0 until nClasses) yield {
      new XYSeries("Class " + String.valueOf(i))
    }).toArray

    val argMax = Nd4j.getExecutioner.exec(new IMax(labels), 1)

    for (i <- 0 until nRows) yield {
      val classIdx = argMax.getDouble(i).toInt
      series(classIdx).add(features.getDouble(i, 0), features.getDouble(i, 1))
    }
    val c = new XYSeriesCollection
    for (s <- series) c.addSeries(s)
    c
  }

  //Test data
  private def createDataSetTest(features: INDArray, labels: INDArray, predicted: INDArray): XYDataset = {
    val nRows = features.rows
    val nClasses = labels.columns

    val series = (for (i <- 0 until nClasses * nClasses) yield {
      val trueClass = i / nClasses
      val predClass = i % nClasses
      val label = "actual=" + trueClass + ", pred=" + predClass
      new XYSeries(label)
    }).toArray

    val actualIdx = Nd4j.getExecutioner.exec(new IMax(labels), 1)
    val predictedIdx = Nd4j.getExecutioner.exec(new IMax(predicted), 1)
    for (i <- 0 until nRows) {
      val classIdx = actualIdx.getDouble(i).toInt
      val predIdx = predictedIdx.getDouble(i).toInt
      val idx = classIdx * nClasses + predIdx
      series(idx).add(features.getDouble(i, 0), features.getDouble(i, 1))
    }
    val c = new XYSeriesCollection
    for (s <- series) c.addSeries(s)
    c
  }

  private def createChart(dataset: XYZDataset, mins: Array[Double], maxs: Array[Double], nPoints: Int, xyData: XYDataset): JFreeChart = {
    val xAxis = new NumberAxis("X")
    xAxis.setRange(mins(0), maxs(0))

    val yAxis = new NumberAxis("Y")
    yAxis.setRange(mins(1), maxs(1))

    val renderer = new XYBlockRenderer
    renderer.setBlockWidth((maxs(0) - mins(0)) / (nPoints - 1))
    renderer.setBlockHeight((maxs(1) - mins(1)) / (nPoints - 1))
    val scale = new GrayPaintScale(0, 1.0)
    renderer.setPaintScale(scale)
    val plot = new XYPlot(dataset, xAxis, yAxis, renderer)
    plot.setBackgroundPaint(Color.lightGray)
    plot.setDomainGridlinesVisible(false)
    plot.setRangeGridlinesVisible(false)
    plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5))
    val chart = new JFreeChart("", plot)
    chart.getXYPlot.getRenderer.setSeriesVisibleInLegend(0, false)

    val scaleAxis = new NumberAxis("Probability (class 0)")
    scaleAxis.setAxisLinePaint(Color.white)
    scaleAxis.setTickMarkPaint(Color.white)
    scaleAxis.setTickLabelFont(new Font("Dialog", Font.PLAIN, 7))
    val legend = new PaintScaleLegend(new GrayPaintScale, scaleAxis)
    legend.setStripOutlineVisible(false)
    legend.setSubdivisionCount(20)
    legend.setAxisLocation(AxisLocation.BOTTOM_OR_LEFT)
    legend.setAxisOffset(5.0)
    legend.setMargin(new RectangleInsets(5, 5, 5, 5))
    legend.setFrame(new BlockBorder(Color.red))
    legend.setPadding(new RectangleInsets(10, 10, 10, 10))
    legend.setStripWidth(10)
    legend.setPosition(RectangleEdge.LEFT)
    chart.addSubtitle(legend)

    ChartUtilities.applyCurrentTheme(chart)

    plot.setDataset(1, xyData)
    val renderer2 = new XYLineAndShapeRenderer
    renderer2.setBaseLinesVisible(false)
    plot.setRenderer(1, renderer2)

    plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD)
    chart
  }

}
