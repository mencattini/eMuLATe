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
class ARL(private val prices: DoubleArray, private val vThreshold: Double, private val sizeWindow: Int) {

    private var delta: Double
    private var eta: Double
    private var rho: Double
    private var x: Double
    private var y: Double
    private var z: Double
    private var weights : DoubleArray
    private var ft: DoubleArray
    private var t: Int
    private var returns: DoubleArray

    init {

        // we check that the sizeWindow is greeter than the prices
        if (sizeWindow > prices.size - 1) {
            val size = prices.size
            throw IllegalArgumentException("sizeWindow=$sizeWindow > prices.size=$size")
        }

        // we init the value with random
        val random = Random()

        delta = random.nextDouble()
        eta = random.nextDouble()
        rho = random.nextDouble()
        x = random.nextDouble()
        y = random.nextDouble()
        z = random.nextDouble()

        // create an array of weight with size of $sizeWindow
        weights = DoubleArray(sizeWindow, {random.nextDouble()})
        returns = DoubleArray(0)

        // the old value
        ft = DoubleArray(0)
        t = 0

        // compute the returns
        computeReturn()
        computeFt()
    }


    /**
     * This function compute the returns and assign it to the returns field.
     */
    private fun computeReturn(){
        returns = DoubleArray(prices.size - 1)
        for (i in 1..(prices.size - 1)){
            returns[i - 1] = prices[i] - prices[i - 1]
        }
    }

    /**
     * Compute the Ft layer using weights, vthresholrd, returns and oldft
     */
    private fun computeFt() : Output {

        // this part doesn't depends on index
        var sum = weights.last() * ft.last() + vThreshold


        // we get the useful weights and returns
        val usefulWeights :DoubleArray
        val usefulReturns :DoubleArray

        // if the t is smaller than our windows, we just take the t first elements
        // we sub 2 because the last weight is used with the F_{t-1}
        if (t < sizeWindow - 2) {
            // as said in the formel neural net layer :
            // w_{i,t} * r_{t-i}
            // so we need to reverse the returns array
            usefulWeights = weights.sliceArray(0..t)
            usefulReturns = returns.sliceArray(0..t).reversedArray()

        } else {
            // we sub the sizeWindow to always get the same number of elements
            // using the maxof(0, t-sizedwindow) to avoid negative index
            // reverse the array for the same thing than above
            usefulWeights = weights.sliceArray(0..(sizeWindow - 2))
            usefulReturns = returns.sliceArray(maxOf(0,t-sizeWindow)..t).reversedArray()
        }

        // we zip the two array together and do the multiplication/sum
        for ((wi, ri) in usefulWeights.zip(usefulReturns)) {
            sum += wi * ri
        }

        when {
            sum > 0.0 -> return Output.Long
            sum < 0.0 -> return Output.Short
            sum == 0.0 -> return Output.Nothing
        }
        return Output.Nothing
    }

    /**
     * Allow us to compute the return. Formule (3) page 3 of article
     *
     *
     */
    private fun computeRt(t: Int) : Double {
        return ft[t-1] * returns[t] - delta * Math.abs(ft[t] - ft[t-1])
    }

    override fun toString(): String {
        return "ARL(returns=${Arrays.toString(returns)}," +
                "\nvThreshold=$vThreshold," +
                "\ndelta=$delta, eta=$eta, rho=$rho, x=$x," + "y=$y, z=$z" +
                "\nweights=${Arrays.toString(weights)}," +
                "\nt=$t)"
    }

}