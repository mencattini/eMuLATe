
import machine.learning.ARL
import java.io.File
import kotlin.collections.ArrayList

fun main(args : Array<String>) {

    // val myFile = File("data/EURUSD_2000_2001_1m_bar.csv").inputStream()
    val myFile = File("data/2006-2010/gbpusd_2006-2010.csv").inputStream()
    val array2 : ArrayList<Double> = arrayListOf()
    myFile.bufferedReader().useLines {
        lines -> lines.forEach {
            array2.add(it.split(";")[1].toDouble())
        }
    }

    val time = System.currentTimeMillis()
    val arl = ARL(20)

    var i = 0
    var p_t = arrayOf(1.0)
    val step = 2000
    val stepLearn = 2500
    //val n = 50000
    val n = 1662000
    val updateThreshold = 200

    arl.initLogging()
    while(i < n) {
        println("$i")
        try {
            arl.train(array2.toDoubleArray().slice(i..i+step), updateThreshold, p_t)
            p_t = arl.test(array2.toDoubleArray().slice(i+step..i+stepLearn), p_t)
            arl.reset()
            if (i % 50000 == 0) {
                arl.saveInFile()
            }
        } catch (_ : ArrayIndexOutOfBoundsException){
            arl.saveInFile()
            break
        }
        i += stepLearn - step
    }

    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
