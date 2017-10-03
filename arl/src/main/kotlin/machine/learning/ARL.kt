package machine.learning

import java.util.*

/**
 *  It's the main class for the Adaptive reinforcement learning
 *
 *  @param arrayPrices the prices data the algorithm will work with. The type is arrayList to get compatibility
 *  with Java.
 *  @param sizeWindow the window size of the element to watch during computation
 *
 *  @author Romain Mencattini
 */
class ARL(private val arrayPrices: List<Double>, private val sizeWindow: Int) {

    private var z: Double
    private var weight : Weights
    private var parameters : Parameters
    private var ft: Array<Pair<Double, Double>> // where first = the sign, second = the value
    private var returns: DoubleArray
    private var prices: DoubleArray

    init {

        val random = Random()

        prices = arrayPrices.toDoubleArray()

        parameters = Parameters()
        z = random.nextDouble()

        // create an array of weight with size of $sizeWindow
        weight = Weights(sizeWindow, 0)
        returns = DoubleArray(0)

        // the old value
        ft = arrayOf(Pair(0.0, 0.0))

    }


    /**
     * In the function we will improve the weights and parameters. This is the learning phase.
     */
    fun trainingLoop(givenT: Int = 1) {

        var t = givenT
        var oldPrice = prices[t - 1]
        // the training is done over every prices. From the soonest to the latest.
        for (price in prices.sliceArray(t..(prices.size - 1))) {

            // compute the return
            val computedReturn = price - oldPrice
            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // compute the Ft
            ft = ft.plus(computeFt(givenT))

            // update the weights
            weight = weight.updateWeights(returns[givenT - 1], ft[givenT - 1].first, ft[givenT].first, givenT,
                    parameters, returns)

            // if the numbers of steps is reach, update the parameters i.e : delta, rho, ...
            val updateThreshold = 1000
            if (t % updateThreshold == 0) {
                parameters = parameters.updateParameters(
                        0.5, 0.5, returns, t - updateThreshold + 1, t, weight, sizeWindow, 10.0)
                println("t=$t")
            }
            // increase the givenT size
            t++
        }
    }

    /**
     * This loop compute the accuracy of prediction for a given array of price.
     *
     * @param prices the prices we want to test the algorithm.
     */
    //TODO fixe the type of prices, it will not work with java
    fun testLoop(givenT: Int = 1, prices : DoubleArray) {

        var t = givenT
        var oldPrice = prices[t - 1]
        // to compute the accuracy of guessing
        var rightGuessed = 0
        var n = 0

        // the training is done over every prices. From the soonest to the latest.
        for (price in prices.sliceArray(t..(prices.size - 1))) {

            // compute the return
            val computedReturn = price - oldPrice

            // compute the number of right guessed sign change.
            if (Math.signum(computedReturn) == ft.last().first) rightGuessed++
            n++

            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // compute the Ft
            ft = ft.plus(computeFt(givenT))

            // update the weights
            weight = weight.updateWeights(returns[givenT - 1], ft[givenT - 1].first, ft[givenT].first, givenT,
                    parameters, returns)

            // increase the givenT size
            t++
        }

        println("accuracy = ${(rightGuessed.toDouble() / n.toDouble()) * 100}")
    }


    /**
     * Compute the Ft layer using weights, vthreshold, returns and old_ft.
     * Given t, my need are :
     * - compute ft
     * - return the value and the sign
     *
     * @param givenT an Int. It's our index.
     * @return a pair of signum and value
     */
    private fun computeFt(givenT : Int) : Pair<Double, Double> {

        return computeFt(givenT, weight, ft, sizeWindow, returns, parameters)
    }

    override fun toString(): String {
        return "ARL(" +
                "parameters=$parameters,\n" +
                "ft=${Arrays.toString(ft)},\n" +
                "returns=${Arrays.toString(returns)}\n," +
                "weigths=" + weight.toString() + ")"
    }


}