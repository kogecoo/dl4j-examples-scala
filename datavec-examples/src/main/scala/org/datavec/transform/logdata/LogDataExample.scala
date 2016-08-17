package org.datavec.transform.logdata

import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils
import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.function.{ Function => SparkFunction }
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader
import org.datavec.api.transform.ReduceOp
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.analysis.DataAnalysis
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.LongColumnCondition
import org.datavec.api.transform.condition.string.StringRegexColumnCondition
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.quality.DataQualityAnalysis
import org.datavec.api.transform.reduce.Reducer
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.AnalyzeSpark
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.joda.time.DateTimeZone

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.URL
import java.util.List
import java.util.zip.GZIPInputStream

import scala.collection.JavaConverters._

/**
 * Simple example performing some preprocessing/aggregation operations on some web log data using DataVec.
 * Specifically:
 * - Load some data
 * - Perform data quality analysis
 * - Perform basic data cleaning and preprocessing
 * - Group records by host, and calculate some aggregate values for each (such as number of requests and total number of bytes)
 * - Analyze the resulting data, and print some results
 *
 *
 * Data is automatically downloaded from: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
 *
 * Examples of some log lines
 * 199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
 * unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3985
 *
 * @author Alex Black
 */
object LogDataExample {

    /** Data URL for downloading */
    final val DATA_URL = "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
    /** Location to save and extract the training/testing data */
    final val DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "datavec_log_example/")
    final val EXTRACTED_PATH = FilenameUtils.concat(DATA_PATH, "data")

    @throws[Exception]
    def main(args: Array[String]) {
        //Setup
        downloadData()
        val conf: SparkConf = new SparkConf()
        conf.setMaster("local[*]")
        conf.setAppName("DataVec Log Data Example")
        val sc: JavaSparkContext = new JavaSparkContext(conf)


        //=====================================================================
        //                 Step 1: Define the input data schema
        //=====================================================================

        //First: let's specify a schema for the data. This is based on the information from: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
        val schema = new Schema.Builder()
            .addColumnString("host")
            .addColumnString("timestamp")
            .addColumnString("request")
            .addColumnInteger("httpReplyCode")
            .addColumnInteger("replyBytes")
            .build()

        //=====================================================================
        //                     Step 2: Clean Invalid Lines
        //=====================================================================

        //Second: let's load the data. Initially as Strings
        var logLines: JavaRDD[String] = sc.textFile(EXTRACTED_PATH)

        //This data unfortunately contains a small number of invalid lines. We'll remove them using standard Spark functionality
        logLines = logLines.filter {
            new SparkFunction[String, java.lang.Boolean]() {
                @throws[Exception]
                override def call(s: String): java.lang.Boolean = {
                    s.matches("(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)");   //Regex for the format we expect
                }
            }
        }

        //=====================================================================
        //         Step 3: Parse Raw Data and Perform Initial Analysis
        //=====================================================================

        //To parse it: we're going to use RegexLineRecordReader. This requires us to define a regex for the format
        val regex = "(\\S+) - - \\[(\\S+ -\\d{4})\\] \"(.+)\" (\\d+) (\\d+|-)"
        val rr = new RegexLineRecordReader(regex,0)
        val parsed = logLines.map(new StringToWritablesFunction(rr))

        //Now, let's check the quality, so we know if there's anything we need to clean up first...
        val dqa = AnalyzeSpark.analyzeQuality(schema, parsed)
        println("----- Data Quality -----")
        println(dqa);    //One issue: non-integer values in "replyBytes" column


        //=====================================================================
        //          Step 4: Perform Cleaning, Parsing and Aggregation
        //=====================================================================

        //Let's specify the transforms we want to do
        val tp = new TransformProcess.Builder(schema)
            //First: clean up the "replyBytes" column by replacing any non-integer entries with the value 0
            .conditionalReplaceValueTransform("replyBytes",new IntWritable(0), new StringRegexColumnCondition("replyBytes","\\D+"))
            //Second: let's parse the date/time string:
            .stringToTimeTransform("timestamp","dd/MMM/YYYY:HH:mm:ss Z", DateTimeZone.forOffsetHours(-4))

            //Group by host and work out summary metrics
            .reduce(new Reducer.Builder(ReduceOp.CountUnique)
                .keyColumns("host")                 //keyColumns == columns to group by
                .countColumns("timestamp")          //Count the number of values
                .countUniqueColumns("request", "httpReplyCode")     //Count the number of unique requests and http reply codes
                .sumColumns("replyBytes")           //Sum the values in the replyBytes column
                .build())

            .renameColumn("count", "numRequests")

            //Finally, let's filter out all hosts that requested less than 1 million bytes in total
            .filter(new ConditionFilter(new LongColumnCondition("sum(replyBytes)", ConditionOp.LessThan, 1000000)))
            .build()

        val executor = new SparkTransformExecutor()
        val processed = executor.execute(parsed, tp)
        processed.cache()


        //=====================================================================
        //       Step 5: Perform Analysis on Final Data; Display Results
        //=====================================================================

        val finalDataSchema = tp.getFinalSchema()
        val finalDataCount = processed.count()
        val sample = processed.take(10)

        val analysis = AnalyzeSpark.analyze(finalDataSchema, processed)

        sc.stop()
        Thread.sleep(4000); //Give spark some time to shut down (and stop spamming console)


        println("----- Final Data Schema -----")
        println(finalDataSchema)

        println("\n\nFinal data count: " + finalDataCount)

        println("\n\n----- Samples of final data -----")
        sample.asScala.foreach(println)

        println("\n\n----- Analysis -----")
        println(analysis)
    }


    @throws[Exception]
    private def downloadData() {
        //Create directory if required
        val directory = new File(DATA_PATH)
        if(!directory.exists()) directory.mkdir()

        //Download file:
        val archivePath = DATA_PATH + "NASA_access_log_Jul95.gz"
        val archiveFile = new File(archivePath)
        val extractedFile = new File(EXTRACTED_PATH,"access_log_July95.txt")
        new File(extractedFile.getParent).mkdirs()

        if( !archiveFile.exists() ){
            println("Starting data download (20MB)...")
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile)
            println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath)
            //Extract tar.gz file to output directory
            extractGzip(archivePath, extractedFile.getAbsolutePath)
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            println("Data (.gz file) already exists at " + archiveFile.getAbsolutePath)
            if( !extractedFile.exists()){
                //Extract tar.gz file to output directory
                extractGzip(archivePath, extractedFile.getAbsolutePath)
            } else {
                println("Data (extracted) already exists at " + extractedFile.getAbsolutePath)
            }
        }
    }

    private final val BUFFER_SIZE = 4096

    @throws[Exception]
    private def extractGzip(filePath: String, outputPath: String) {
        println("Extracting files...")
        val buffer: Array[Byte] = new Array[Byte](BUFFER_SIZE)

        try{
            val gzis = new GZIPInputStream(new FileInputStream(new File(filePath)))

            val out = new FileOutputStream(new File(outputPath))

            var len: Int = gzis.read(buffer)
            while (len > 0) {
                out.write(buffer, 0, len)
                len = gzis.read(buffer)
            }

            gzis.close()
            out.close()

            println("Done")
        } catch { case ex: IOException => ex.printStackTrace() }
    }

}
