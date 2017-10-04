package machine.learning

import java.util.*

/**
 *  It's the main class for the Adaptive reinforcement learning
 *
 *  @param sizeWindow the window size of the element to watch during computation
 *
 *  @author Romain Mencattini
 */
class ARL(private val sizeWindow: Int) {

    private var z: Double
    private var weight : Weights
    private var parameters : Parameters
    private var ft: Array<Pair<Double, Double>> // where first = the sign, second = the value
    private var returns: DoubleArray

    init {

        val random = Random()

        parameters = Parameters()
        z = random.nextDouble()

        // create an array of weight with size of $sizeWindow
        weight = Weights(sizeWindow, 0)
        returns = DoubleArray(0)

        // the old value
        ft = arrayOf(Pair(0.0, 0.0))

    }


    /**
     * This loop compute the accuracy of prediction for a given array of price.
     *
     * @param prices the prices we want to test the algorithm.
     * @param test the boolean that said if we want to test or to train. false => train loop, true => test loop
     * @param updateThreshold the number of test before update of the parameters
     */
    fun loop(prices : List<Double>, test: Boolean = false, updateThreshold: Int = 1000) {

        val pricesCasted = prices.toDoubleArray()
        var t = 1
        var oldPrice = pricesCasted[t - 1]
        // to compute the accuracy of guessing
        var rightGuessed = 0
        var n = 0

        // the training is done over every pricesCasted. From the soonest to the latest.
        for (price in pricesCasted.sliceArray(t..(pricesCasted.size - 1))) {

            // compute the return
            val computedReturn = price - oldPrice

            // compute the number of right guessed sign change.
            // if we do nothing (F(t) = 0), we don't to judge that as a failure.
            // we do this test only if the test == true
            if (test && (Math.signum(computedReturn) == ft.last().first || computedReturn == 0.0)) rightGuessed++
            n++

            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // compute the Ft
            ft = ft.plus(computeFt(t))

            // update the weights
            weight = weight.updateWeights(returns[t - 1], ft[t - 1].first, ft[t].first, t,
                    parameters, returns)

            // if the numbers of steps is reach, update the parameters i.e : delta, rho, ...
            if (t % updateThreshold == 0) {
                parameters = parameters.parallelUpdateParameters(
                        0.5, 0.5, returns.sliceArray((t - updateThreshold + 1)..t)
                        , weight, sizeWindow, 1.0)
                println("t=$t")
            }
            // increase the givenT size
            t++
        }

        // we print only if it's in test
        if (test) println("accuracy = ${(rightGuessed.toDouble() / n.toDouble()) * 100}")
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

        return computeFt(givenT, this.weight, this.ft, this.sizeWindow, this.returns, this.parameters)
    }

    override fun toString(): String {
        return "ARL(" +
                "parameters=$parameters,\n" +
                "weights=" + weight.toString() + ")"
    }


}