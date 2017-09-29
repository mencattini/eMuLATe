import machine.learning.ARL;
import java.io.File

fun main(args : Array<String>) {

    // we open the .dat file located in data
    val myFile =  File("data/EURUSD.dat").inputStream()
    var array2 : ArrayList<Double> = arrayListOf()

    myFile.bufferedReader().useLines { lines -> lines.forEach {
        array2.add( it.split("/")[0].split(" ").last().toDouble())
    } }

    array2.toDoubleArray().slice(0..2000)
    val array = arrayListOf<Double>(1.99, 1.88, 1.77, 1.66, 1.55, 1.66, 1.77, 1.88, 1.77, 1.66, 1.55, 1.44, 1.33, 1.22, 1.11, 1.00)


    var arl = ARL(array2.toDoubleArray().slice(0..10000), 4)
    arl.trainingLoop()
    println(arl.toString())
}