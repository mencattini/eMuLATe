import machine.learning.ARL

fun main(args : Array<String>) {

    val array : DoubleArray = doubleArrayOf(1.99, 1.88, 1.77, 1.66, 1.55, 1.44, 1.33, 1.22)
    var arl = ARL(array, 0.0, 3)
}