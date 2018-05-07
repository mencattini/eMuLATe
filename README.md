# eMuLATe
Machine Learning Automatic Trading

## Example :

We work on 300000 minutes of the EURUSD market. The best step are 2000 $L_{train}$ and 500 $L_{test}$. At the end, we print the cumulative profit and the value for the market.

```kotlin
import machine.learning.ARL
import java.io.File
import kotlin.collections.ArrayList

fun main(args : Array<String>) {

    // fx github file
    /*val myFile = File("data/2006-2010/eurusd-2006_2010-days.csv").inputStream()
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

    arl.initLogging()
    // backtesting loop
    while(i < n) {
        println("$i")
        arl.train(array2.toDoubleArray().slice(i..i+step), updateThreshold, p_t)
        p_t = arl.test(array2.toDoubleArray().slice(i+step..i+stepLearn), p_t)
        arl.reset()
        if (i % 10000 == 0) {
            arl.saveInFile()
        }
        i += stepLearn - step
    }
    arl.saveInFile()

    println("time = ${(System.currentTimeMillis() - time) / 1000}")

}
```
