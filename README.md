# eMuLATe
Machine Learning Automatic Trading

## To correct :
* implemente the 3rd layer.

## Example :

We work on 100000 ticks of the EURUSD market. The best step are 2000 L_train and 500 L_test. At the end, we print the cumulative profit and the value for the market.

```kotlin
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
    val arl = ARL(20)

    var i = 0
    var p_t = arrayOf(1.0)
    val step = 2000
    val stepLearn = 2500
    val n = 100000
    val updateThreshold = 200

    while(i < n) {
        println("$i")
        arl.train(array2.toDoubleArray().slice(i..i+step), updateThreshold, p_t)
        p_t = arl.test(array2.toDoubleArray().slice(i+step..i+stepLearn), p_t)
        arl.reset()
        i += stepLearn - step
    }

    figure(1)
    plot(p_t.toDoubleArray())
    xlabel("ticks")
    ylabel("p_t")
    title("Run")

    figure(2)
    plot(array2.slice(0..n).toDoubleArray())
    xlabel("ticks")
    ylabel("value")
    title("EURUSD")


    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
```
