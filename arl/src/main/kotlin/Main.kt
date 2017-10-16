
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
            array2.add(it.split("/")[0].split(" ").last().toDouble())
        }
    }

    val time = System.currentTimeMillis()
    val arl = ARL(10)

    var i = 100000
    val copyI = i
    var p_t = arrayOf(0.0)
    val step = 2000
    val n = 200000

    while(i < n) {
        println("$i")
        arl.loop(array2.toDoubleArray().slice(i..i+step), 100)
        p_t = arl.loop(array2.toDoubleArray().slice(i+step..i+step+500), 100, p_t)
        arl.reset()
        i += step

    }

    figure(1)
    plot(p_t.toDoubleArray())
    xlabel("ticks")
    ylabel("p_t")
    title("Run")

//    figure(2)
//    plot(array2.slice(i..n).toDoubleArray())
//    xlabel("ticks")
//    ylabel("value")
//    title("EURUSD")


    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
