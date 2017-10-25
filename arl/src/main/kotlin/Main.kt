
import machine.learning.ARL
import java.io.File
import java.util.*
import koma.*

fun main(args : Array<String>) {

    // we open the .dat file located in data
    val myFile = File("data/EURUSD.dat").inputStream()
    val array2: ArrayList<Double> = arrayListOf()

    myFile.bufferedReader().useLines { lines ->
        lines.forEach {
            array2.add(it.split("/")[1].split(" ")[0].toDouble())
        }
    }

    val time = System.currentTimeMillis()
    val arl = ARL(10)

    var i = 0
    val copyI = i
    var p_t = arrayOf(0.0)
    val step = 2000
    val stepLearn = 2500
    val n = 100000

    while(i < n) {
        println("$i")
        arl.loop(array2.toDoubleArray().slice(i..i+step), true,200)
        p_t = arl.loop(array2.toDoubleArray().slice(i+step..i+stepLearn), false,200, p_t)
        arl.reset()
        i += stepLearn - step
    }

    figure(1)
    plot(p_t.toDoubleArray())
    xlabel("ticks")
    ylabel("p_t")
    title("Run")

//    figure(2)
//    plot(array2.slice(copyI..n).toDoubleArray())
//    xlabel("ticks")
//    ylabel("value")
//    title("EURUSD")


    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
