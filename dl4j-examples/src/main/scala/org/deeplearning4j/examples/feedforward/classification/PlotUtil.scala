package org.deeplearning4j.examples.feedforward.classification

import java.awt._
import javax.swing._

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
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax
import org.nd4j.linalg.factory.Nd4j

/**Simple plotting methods for the MLPClassifier examples
 * @author Alex Black
 */
object PlotUtil {

    /**Plot the training data. Assume 2d input, classification output
     * @param features Training data features
     * @param labels Training data labels (one-hot representation)
     * @param backgroundIn sets of x,y points in input space, plotted in the background
     * @param backgroundOut results of network evaluation at points in x,y points in space
     * @param nDivisions Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    def plotTrainingData(features: INDArray, labels: INDArray, backgroundIn: INDArray, backgroundOut: INDArray, nDivisions: Int): Unit = {
        val mins: Array[Double] = backgroundIn.min(0).data().asDouble()
        val maxs: Array[Double] = backgroundIn.max(0).data().asDouble()

        val backgroundData = createBackgroundData(backgroundIn, backgroundOut)
        val panel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTrain(features, labels)))

        val f: JFrame = new JFrame()
        f.add(panel)
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        f.pack()
        f.setTitle("Training Data")

        f.setVisible(true)
    }

    /**Plot the training data. Assume 2d input, classification output
     * @param features Training data features
     * @param labels Training data labels (one-hot representation)
     * @param predicted Network predictions, for the test points
     * @param backgroundIn sets of x,y points in input space, plotted in the background
     * @param backgroundOut results of network evaluation at points in x,y points in space
     * @param nDivisions Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    def plotTestData(features: INDArray, labels: INDArray, predicted: INDArray, backgroundIn: INDArray, backgroundOut: INDArray, nDivisions: Int): Unit = {

        val mins = backgroundIn.min(0).data().asDouble()
        val maxs = backgroundIn.max(0).data().asDouble()

        val backgroundData: XYZDataset = createBackgroundData(backgroundIn, backgroundOut)
        val panel: JPanel = new ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTest(features, labels, predicted)))

        val f: JFrame = new JFrame()
        f.add(panel)
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
        f.pack()
        f.setTitle("Test Data")

        f.setVisible(true)

    }


    /**Create data for the background data set
     */
    private[this] def createBackgroundData(backgroundIn: INDArray, backgroundOut: INDArray): XYZDataset = {
        val nRows = backgroundIn.rows()
        val (xValues, yValues, zValues) = (0 until nRows).map({ i =>
            val xValue = backgroundIn.getDouble(i,0)
            val yValue = backgroundIn.getDouble(i,1)
            val zValue = backgroundOut.getDouble(i)
            (xValue, yValue, zValue)
        }).unzip3
        val series: Array[Array[Double]] = Array(xValues.toArray, yValues.toArray, zValues.toArray)

        val dataset: DefaultXYZDataset = new DefaultXYZDataset()
        dataset.addSeries("Series 1", series)
        return dataset
    }

    //Training data
    private[this] def createDataSetTrain(features: INDArray, labels: INDArray ): XYDataset = {
        val nRows = features.rows()

        val nClasses = labels.columns()

        val series: Array[XYSeries] = (0 until nClasses).map({ i =>
            new XYSeries("Class " + String.valueOf(i))
        }).toArray
        val argMax: INDArray = Nd4j.getExecutioner().exec(new IAMax(labels), 1)
        (0 until nRows).foreach { i =>
            val classIdx = argMax.getDouble(i).toInt
            series(classIdx).add(features.getDouble(i, 0), features.getDouble(i, 1))
        }

        val c = new XYSeriesCollection()
        series.foreach(c.addSeries)
        c
    }

    //Test data
    private def createDataSetTest(features: INDArray, labels: INDArray, predicted: INDArray): XYDataset = {
        val nRows = features.rows()

        val nClasses = labels.columns()

        val series: Array[XYSeries] = (0 until nClasses * nClasses).map ({ i =>
            val trueClass = i/nClasses
            val predClass = i%nClasses
            val label = "actual=" + trueClass + ", pred=" + predClass
            new XYSeries(label)
        }).toArray
        val actualIdx = Nd4j.getExecutioner().exec(new IAMax(labels), 1)
        val predictedIdx = Nd4j.getExecutioner().exec(new IAMax(predicted), 1)
        (0 until nRows).foreach { i =>
            val classIdx = actualIdx.getDouble(i).toInt
            val predIdx = predictedIdx.getDouble(i).toInt
            val idx = classIdx * nClasses + predIdx
            series(idx).add(features.getDouble(i, 0), features.getDouble(i, 1))
        }

        val c = new XYSeriesCollection()
        series.foreach(c.addSeries)
        c
    }

    private def createChart(dataset: XYZDataset , mins: Array[Double], maxs: Array[Double], nPoints: Int, xyData: XYDataset): JFreeChart = {
        val xAxis = new NumberAxis("X")
        xAxis.setRange(mins(0), maxs(0))


        val yAxis = new NumberAxis("Y")
        yAxis.setRange(mins(1), maxs(1))

        val renderer = new XYBlockRenderer()
        renderer.setBlockWidth((maxs(0)-mins(0))/(nPoints-1))
        renderer.setBlockHeight((maxs(1) - mins(1)) / (nPoints - 1))
        val scale = new GrayPaintScale(0, 1.0)
        renderer.setPaintScale(scale)
        val plot = new XYPlot(dataset, xAxis, yAxis, renderer)
        plot.setBackgroundPaint(Color.lightGray)
        plot.setDomainGridlinesVisible(false)
        plot.setRangeGridlinesVisible(false)
        plot.setAxisOffset(new RectangleInsets(5, 5, 5, 5))
        val chart = new JFreeChart("", plot)
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false)


        val scaleAxis = new NumberAxis("Probability (class 0)")
        scaleAxis.setAxisLinePaint(Color.white)
        scaleAxis.setTickMarkPaint(Color.white)
        scaleAxis.setTickLabelFont(new Font("Dialog", Font.PLAIN, 7))
        val legend = new PaintScaleLegend(new GrayPaintScale(),
                scaleAxis)
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
        val renderer2 = new XYLineAndShapeRenderer()
        renderer2.setBaseLinesVisible(false)
        plot.setRenderer(1, renderer2)

        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD)

        chart
    }

}
