package machine.learning

import java.util.*

/**
 *  It's the main classe for the Adaptive reinforcement learning
 *
 *  @param prices the prices data the algorithm will work with
 *  @param vThreshold the threshold of the neural net
 *  @param sizeWindow the window size of the element to watch during computation
 *
 *  @author Romain Mencattini
 */
class ARL(private var prices: DoubleArray, private val vThreshold: Double, private val sizeWindow: Int) {

    private var z: Double
    private var weights : Array<Weights>
    private var parameters : Parameters
    private var ft: Array<Pair<Double, Double>>
    private var returns: DoubleArray

    init {

        val random = Random()

        parameters = Parameters()
        z = random.nextDouble()

        // create an array of weight with size of $sizeWindow
        weights = arrayOf(Weights(sizeWindow, 0))
        returns = DoubleArray(0)

        // the old value
        ft = arrayOf(Pair(0.0, 0.0))

    }

    /**
     * It will reset the returns and the prices.
     * It's useful between two runs to keep the weights and the parameters but not the rest.
     */
    public fun reset() {
        prices = DoubleArray(0)
        returns = DoubleArray(0)
    }

    /**
     * It sets the prices for a new runs
     */
    public fun setPrices(prices: DoubleArray) {
        this.prices = prices
    }

    /**
     * In the function we will improve the weights and parameters. This is the learning phase.
     */
    public fun mainLoop() {

        var oldPrice = prices[0]
        var givenT = 0
        // the training is done over every prices. From the soonest to the latest.
        for (price in prices.sliceArray(1..(prices.size - 1))) {

            // compute the return
            var computedReturn = price - oldPrice
            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // compute the Ft

            // update the weights

            // if the numbers of steps is reach, update the parameters i.e : delta, rho, ...
        }
    }

    override fun toString(): String {
        return "ARL(returns=${Arrays.toString(returns)})"
    }


    /**
//     * Compute the Ft layer using weights, vthresholrd, returns and old_ft.
//     * Given t, my need are :
//     * - compute ft
//     * - return the value and the sign
//     *
//     * @param givenT an Int. It's our index.
//     * @return a pair of value and signum
//     */
//    private fun computeFt(givenT : Int) : Pair<Double, Double> {
//
//        // if the t > returns.size, we are out of the array and we
//        // needs more samples
//        when {
//            givenT > returns.size - 1 ->
//                throw Exception("t($givenT) is greeter than returns.size(" + returns.size.toString() + ").")
//            givenT > ft.size ->
//                throw Exception("t($givenT) is greeter than ft.size(" + ft.size.toString() + ").")
//        }
//
//        // this part doesn't depends on index
//        var sum = weights[givenT] * ft[givenT - 1] + vThreshold
//
//
//        // we get the useful weights and returns
//        val usefulWeights :DoubleArray
//        val usefulReturns :DoubleArray
//
//        // if the t is smaller than our windows, we just take the t first elements
//        // we sub 2 because the last weight is used with the F_{t-1}
//        if (givenT < sizeWindow - 2) {
//            // as said in the formel neural net layer :
//            // w_{i,t} * r_{t-i}
//            // so we need to reverse the returns array
//            usefulWeights = weights.sliceArray(0..givenT)
//            usefulReturns = returns.sliceArray(0..givenT).reversedArray()
//
//        } else {
//            // we sub the (sizeWindow - 2) to always get the same number of elements than the weights
//            // using the maxof(0, t-sizedwindow + 2) to avoid negative index
//            // reverse the array for the same thing than above
//            usefulWeights = weights.sliceArray(0..(sizeWindow - 2))
//            usefulReturns = returns.sliceArray(maxOf(0,givenT-sizeWindow + 2)..givenT).reversedArray()
//        }
//
//        // we zip the two array together and do the multiplication/sum
//        for ((wi, ri) in usefulWeights.zip(usefulReturns)) {
//            sum += wi * ri
//        }
//
//        // we add the result to the ft vector
//        ft[givenT - 1] = Math.signum(sum)
//
//        return Pair(sum, Math.signum(sum))
//    }



}