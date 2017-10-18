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
    private var ft: Array<Double> // each element is the result of Math.signum(x)
    private var returns: DoubleArray
    private var positionProfit :PositionProfit

    init {

        val random = Random()

        parameters = Parameters()
        z = random.nextDouble()

        // create an array of weight with size of $sizeWindow
        weight = Weights(sizeWindow, 0)
        returns = DoubleArray(0)

        // the old value
        ft = arrayOf(Math.signum(0.0))

        positionProfit = PositionProfit(0.0,0.0)
    }


    /**
     * This loop compute the accuracy of prediction for a given array of price.
     *
     * @param prices the prices we want to test the algorithm.
     * @param updateThreshold the number of test before update of the parameters
     * @param oldPt the array of cumulative profit
     *
     * @return we return the w_n (cf. article to the meaning and computation)
     */
    fun loop(prices : List<Double>, train : Boolean ,updateThreshold: Int = 1000,oldPt: Array<Double> = arrayOf(0.0)): Array<Double> {

        // we cast the price, for the compatibility with java
        val pricesCasted = prices.toDoubleArray()
        var t = 1
        var oldPrice = pricesCasted[t - 1]

        // we compute the p_t, it's an array
        var pt = oldPt.clone()
        positionProfit.currentProfit = pt.last()
        positionProfit.lastPositionProfit = pt.last()

        // the training is done over every pricesCasted. From the soonest to the latest.
        for (price in pricesCasted.sliceArray(t..(pricesCasted.size - 1))) {

            // compute the return
            val computedReturn = price - oldPrice

            // keep the price for the next loop
            oldPrice = price
            // store the computedReturn in returns
            returns = returns.plus(computedReturn)

            // update the current profit
            positionProfit.currentProfit = pt.last()

            // compute the Ft
            val firstComputedFt = computeFt(t)

            // update the weights
            weight = weight.updateWeights(returns[t - 1], ft[t - 1], firstComputedFt, t,
                    parameters, returns)

            // cf. article, "since the weight updating is designed to improve the model at each step, it makes sense to
            // recalculate the trading decision with the most up-to-date version [...] This final trading signal is
            // used for effective decision making by the risk and the performance control layer."
            ft = ft.plus(computeFt(t, firstComputedFt))

            // if the numbers of steps is reach, update the parameters i.e : delta, rho, ...
            if (train && t % updateThreshold == 0) {
                parameters = parameters.parallelUpdateParameters(
                        0.1, 0.5, returns.sliceArray((t - updateThreshold + 1)..t)
                        ,weight, sizeWindow, 1.0)
            }
            // increase the givenT size
            t++

            // we compute the w_n
            val lastIndex = ft.lastIndex
            // R_t := F_{t-1} r_t - delta |F_{t} - F_{t-1}|
            pt = pt.plus( pt.last() +
                    (ft[lastIndex - 1] * returns.last() - parameters.delta *
                            Math.abs(ft[lastIndex] - ft[lastIndex - 1])))

        }

        // return the p_t
        return pt
    }

    /**
     * Reset the weight and the returns between runs.
     */
    fun reset() {
        this.weight = Weights(sizeWindow, 0)
        this.returns = DoubleArray(0)
        this.ft = arrayOf(Math.signum(0.0))
        this.parameters = Parameters()
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
    private fun computeFt(givenT : Int, oldFt :Double = ft.last()) : Double {

        return computeFt(givenT, this.weight, oldFt, this.sizeWindow,
                this.returns, this.parameters, this.positionProfit)
    }

    override fun toString(): String {
        return "ARL(" +
                "parameters=$parameters,\n" +
                "weights=" + weight.toString() + ")"
    }


}