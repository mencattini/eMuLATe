package machine.learning

import java.util.*
import computeFt

/**
 * @author Romain Mencattini
 *
 */
class Parameters {

    var delta : Double
    var eta : Double
    var rho : Double
    var x : Double
    var y : Double

    init {
        // we init the value with random
        val random = Random()

        delta = random.nextDouble()
        eta = random.nextDouble()
        rho = random.nextDouble()
        x = random.nextDouble()
        y = random.nextDouble()
    }

    /**
     * A custom constructor.
     *
     * @param delta : the new delta
     * @param eta : the new eta
     * @param rho : the new rho
     * @param x : the new x
     * @param y : the new y
     *
     * @return a new Parameters object with custom value for every fields.
     */
    fun Parameters(delta: Double, eta: Double, rho: Double, x : Double, y : Double): Parameters {

        val param = Parameters()
        param.delta = delta
        param.eta = eta
        param.rho = rho
        param.x = x
        param.y = y

        return param
    }

    // TODO : add the java doc
    fun costFunction(a: Double, v: Double, returns: DoubleArray,
                         startT: Int, endT: Int, weight: Weights,
                         sizeWindow: Int) : Double {

        val parameters = Parameters()

        val rt = getRt(returns, startT, endT, parameters, weight, sizeWindow)
        val sumRtPositive = rt.filter { it < 0 }.map { it -> it * it }.sum()
        val sumRtNegative = rt.filter { it > 0 }.map { it -> it * it }.sum()
        val sigma = sumRtNegative / sumRtPositive

        // TODO : implemente the real W_N / N
        val rbar = rt.sum() / rt.size
        return a * ( 1 - v ) * rbar - v * sigma
    }

    /**
     * This function compute the R_t := F_{t-1} * r_t - delta * | F_t - F_{t-1}| with custom parameters.
     *
     * @param returns the array with the computed returns
     * @param startT the index to start
     * @param endT the index to stop
     * @param parameters the custom parameters
     * @param weight the weights of the neural net
     * @param sizeWindow the number of considered elements
     *
     * @return a array of double containing the R_t
     */
    private fun getRt(returns: DoubleArray, startT: Int, endT: Int, parameters: Parameters, weight: Weights,
              sizeWindow: Int): DoubleArray {

        // it's our index to iterate through the array.
        var t = startT
        // the array we will return
        val rt = DoubleArray(endT - startT)
        var ft = arrayOf(Pair(0.0, 0.0))
        var mutableWeight = weight.copy()

        while (t < endT) {
            // we compute the ft
            ft = ft.plus(computeFt(t, mutableWeight, ft, sizeWindow, returns, parameters))
            // update the weights
            mutableWeight = mutableWeight.updateWeights(t, parameters, ft, returns)
            // store the result
            rt[t - startT] = ft[t - 1].first * returns[t] - delta * Math.abs(ft[t].first - ft[t - 1].first)

            t++
        }

        return rt
    }


    // TODO : implemente the updateParameters function

    override fun toString(): String {
        return "Parameters(delta=$delta, eta=$eta, rho=$rho, x=$x, y=$y)"
    }


}