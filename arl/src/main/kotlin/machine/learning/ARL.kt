package machine.learning

import java.util.*
import kotlin.collections.ArrayList

/**
 *  It's the main classe for the Adaptive reinforcement learning
 *
 *  @param prices the prices data the algorithm will work with
 *  @param vThreshold the threshold of the neural net
 *  @param sizeWindow the window size of the element to watch during computation
 *
 *  @author Romain Mencattini
 */
class ARL(private val arrayPrices: ArrayList<Double>, private val vThreshold: Double, private val sizeWindow: Int) {

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
    public fun trainingLoop() {

        var oldPrice = prices[0]
        var givenT = 1
        // the training is done over every prices. From the soonest to the latest.
        for (price in prices.sliceArray(1..(prices.size - 1))) {

            // compute the return
            var computedReturn = price - oldPrice
            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // compute the Ft
            ft = ft.plus(computeFt(givenT))

            // update the weights
            weight = weight.updateWeights(givenT, parameters, ft, returns)

            // if the numbers of steps is reach, update the parameters i.e : delta, rho, ...

            // increase the givenT size
            givenT++
        }
    }


    /**
     * Compute the Ft layer using weights, vthresholrd, returns and old_ft.
     * Given t, my need are :
     * - compute ft
     * - return the value and the sign
     *
     * @param givenT an Int. It's our index.
     * @return a pair of signum and value
     */
    private fun computeFt(givenT : Int) : Pair<Double, Double> {

        // this part doesn't depends on index
        var sum = weight.coefficients.last() * ft[givenT - 1].first + vThreshold


        // we get the useful weights and returns
        val usefulWeights :DoubleArray
        val usefulReturns :DoubleArray

        // if the t is smaller than our windows, we just take the t first elements
        // we sub 2 because the last weight is used with the F_{t-1}
        if (givenT < sizeWindow - 2) {
            // as said in the formal neural net layer :
            // w_{i,t} * r_{t-i}
            // so we need to reverse the returns array
            usefulWeights = weight.coefficients.sliceArray(0..givenT)
            usefulReturns = returns.sliceArray(0..givenT).reversedArray()

        } else {
            // we sub the (sizeWindow - 2) to always get the same number of elements than the weights
            // using the maxof(0, t-sizedwindow + 2) to avoid negative index
            // reverse the array for the same thing than above
            usefulWeights = weight.coefficients.sliceArray(0..(sizeWindow - 2))
            usefulReturns = returns.sliceArray(maxOf(0,givenT-sizeWindow + 2)..givenT).reversedArray()
        }

        // we zip the two array together and do the multiplication/sum
        for ((wi, ri) in usefulWeights.zip(usefulReturns)) {
            sum += wi * ri
        }

        // we reutrn the result to the ft vector
        return Pair(Math.signum(sum), sum)
    }

    override fun toString(): String {
        return "ARL(" +
                "parameters=$parameters,\n" +
                "ft=${Arrays.toString(ft)},\n" +
                "returns=${Arrays.toString(returns)}\n," +
                "weigths=${Arrays.toString(weight.coefficients)}" +
                ")"
    }


}