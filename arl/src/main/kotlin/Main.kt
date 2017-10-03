
import machine.learning.ARL
import java.io.File
import java.util.*

fun main(args : Array<String>) {

    // we open the .dat file located in data
    val myFile =  File("data/EURUSD.dat").inputStream()
    val array2 : ArrayList<Double> = arrayListOf()

    myFile.bufferedReader().useLines { lines -> lines.forEach {
        array2.add( it.split("/")[0].split(" ").last().toDouble())
    } }

    val time = System.currentTimeMillis()
    val arl = ARL(array2.toDoubleArray().slice(0..100000), 5)

    arl.trainingLoop()
    arl.testLoop(prices=array2.toDoubleArray().slice(50000..60000).toDoubleArray())

//    println(arl.toString())
}