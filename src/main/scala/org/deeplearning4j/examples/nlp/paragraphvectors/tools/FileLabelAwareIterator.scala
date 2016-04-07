package org.deeplearning4j.examples.nlp.paragraphvectors.tools

import java.io.{BufferedReader, File, FileReader}
import java.util.concurrent.atomic.AtomicInteger

import org.deeplearning4j.text.documentiterator.{LabelAwareIterator, LabelledDocument, LabelsSource}

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 * This is simple filesystem-based LabelAware iterator.
 * It assumes that you have one or more folders organized in the following way:
 * 1st level subfolder: label name
 * 2nd level: bunch of documents for that label
 *
 * You can have as many label folders as you want, as well.
 *
 * Please note: as of DL4j 3.9 this iterator is available as part of DL4j codebase, so there's no need to use this implementation.
 *
 * @author raver119@gmail.com
 */
class FileLabelAwareIterator protected(protected val files: java.util.List[File], protected val labelsSource: LabelsSource) extends LabelAwareIterator {
    protected val position = new AtomicInteger(0)

    /*
        Please keep this method protected, it's used in tests
     */
    protected def this() = this(null, null)

    override def hasNextDocument: Boolean = position.get() < files.size()

    override def nextDocument(): LabelledDocument = {
        val fileToRead: File = files.get(position.getAndIncrement())
        val label: String = fileToRead.getParentFile.getName
        try {
            val document = new LabelledDocument()
            val reader = new BufferedReader(new FileReader(fileToRead))
            val builder = new StringBuilder()
            var line = reader.readLine()
            while (line != null) {
                builder.append(line).append(" ")
                line = reader.readLine()
            }

            document.setContent(builder.toString())
            document.setLabel(label)

            document
        } catch  {
            case e: Exception => throw new RuntimeException(e)
        }
    }

    override def reset() {
        position.set(0)
    }

    override def getLabelsSource: LabelsSource = labelsSource

}


object FileLabelAwareIterator {
    class Builder() {
        protected val foldersToScan = mutable.ArrayBuffer.empty[File]

        /**
         * Root folder for labels -> documents.
         * Each subfolder name will be presented as label, and contents of this folder will be represented as LabelledDocument, with label attached
         *
         * @param folder folder to be scanned for labels and files
         * @return
         */
        def addSourceFolder(folder: File): Builder = {
            foldersToScan += folder
            this
        }

        def build(): FileLabelAwareIterator = {
            // search for all files in all folders provided
            val fileList = mutable.ArrayBuffer.empty[File]
            val labels = mutable.ArrayBuffer.empty[String]

            foldersToScan.foreach { case (file: File) if file.isDirectory =>
                val files = file.listFiles()
                if (files != null && files.nonEmpty) {
                    files.foreach { case fileLabel if fileLabel.isDirectory =>
                        if (!labels.contains(fileLabel.getName)) labels += fileLabel.getName

                        val docs = fileLabel.listFiles()
                        if (docs != null && docs.nonEmpty) {
                            docs.foreach { case fileDoc if !fileDoc.isDirectory => fileList += fileDoc }
                        }
                    }
                }
            }
            val source = new LabelsSource(labels.toList.asJava)
            val iterator = new FileLabelAwareIterator(fileList.toList.asJava, source)

            iterator
        }
    }
}