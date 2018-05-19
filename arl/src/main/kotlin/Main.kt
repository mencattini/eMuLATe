
import machine.learning.ARL
import java.io.File
import kotlin.collections.ArrayList

fun main(args : Array<String>) {

    // fx github file
    /*val myFile = File("data/2000-2001/EURUSD_2000-2001.csv").inputStream()
    val array2 : ArrayList<Double> = arrayListOf()
    myFile.bufferedReader().useLines {
        lines -> lines.forEach {
            array2.add(it.split(";")[1].toDouble())
        }
    }*/
    // duka github file
    val myFile = File("data/2004-2005/EURUSD-2004_01_01-2005_01_01.csv").inputStream()
    val array2 : ArrayList<Double> = arrayListOf()
        myFile.bufferedReader().useLines {
            lines -> lines.forEach {
            array2.add(it.split(",")[1].toDouble())
        }
    }

    val time = System.currentTimeMillis()
    val arl = ARL(20)

    var i = 0
    var p_t = arrayOf(1.0)
    val step = 2000
    val stepLearn = 2500
    val n = 300000

    val updateThreshold = 200

    // naive mean
    // np.mean(sharpe) ~= 70; np.median(sharpe) ~= 60
    var mean = 70.0
    var meanCounting = 1

    arl.initLogging()
    // backtesting loop
    while(i < n) {
        println("$i")
        // get the sharpe ratio from training
        val sharpe = arl.train(array2.toDoubleArray().slice(i..i + step), updateThreshold, p_t)
        // compute the mean of positive sharpe ratio
        // online mean updating : http://www.heikohoffmann.de/htmlthesis/node134.html
        mean = if (sharpe > 0.0) mean + (1/ meanCounting + 1) * (sharpe - mean) else mean
        meanCounting++
        // apply the test only if the sharpe ratio is above the mean.
        p_t = if (sharpe >= mean) {
            arl.test(array2.toDoubleArray().slice(i + step..i + stepLearn), p_t)
        } else {
            // else we just stay in position 0.
            arl.savedPt = arl.savedPt.plus(DoubleArray(stepLearn - step, {p_t.last()}))
            arl.savedFt = arl.savedFt.plus(DoubleArray(stepLearn - step, {0.0}))
            DoubleArray(stepLearn - step, {p_t.last()}).toTypedArray()
        }
        arl.reset()
        if (i % 10000 == 0) {
            arl.saveInFile()
        }
        i += stepLearn - step
    }
    arl.saveInFile()
    println("mean = $mean")
    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
