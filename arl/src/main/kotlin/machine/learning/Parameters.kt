package machine.learning

import org.apache.commons.math3.distribution.NormalDistribution
import java.util.concurrent.Executors

/**
 * @author Romain Mencattini
 *
 * This class centralize the parameters used in the first and second layer.
 *
 */
internal class Parameters {

    var delta: Double = 0.001
    var eta: Double = 0.00001
    private var rho: Double = 0.01
    var x: Double = 0.005
    var y: Double = 0.0001

    @Volatile private lateinit var best: Pair<Double, Parameters>
    private val epsilon = 1e-8


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
    private fun Parameters(delta: Double, eta: Double, rho: Double, x: Double, y: Double): Parameters {

        val param = Parameters()
        param.delta = delta
        param.eta = eta
        param.rho = rho
        param.x = x
        param.y = y

        return param
    }


    /**
     * The function we want to optimize.
     *
     * @param a cf. article, it's a fixed parameters
     * @param v cf. article, it's a fixed parameters
     * @param returns the array of r_t
     * @param weight the weights
     * @param sizeWindow the number of considered elements
     *
     * @return the result of the cost function
     */
    private fun costFunction(a: Double, v: Double, returns: DoubleArray, weight: Weights, sizeWindow: Int): Double {

        val rt = getRt(returns,this, weight, sizeWindow)
        var sumRtNegative = 0.0
        var sumRtPositive = 0.0
//        filter the positive and negative element
        for (ele in rt) {
            if (ele > 0.0) {
                sumRtPositive += ele
            } else if (ele < 0.0) {
                sumRtNegative += ele
            }
        }
        // check if there is a Nan or not, if Nan we just return the neutral element of multiplication
        // we check the denominator
        val sigma = if (sumRtPositive == 0.0) Double.MAX_VALUE else sumRtNegative / sumRtPositive

        val wN = rt.sum() / rt.size
        // return the result or the neutral element of multiplication
        // we check the denominator
        val rBar = if (wN == Double.NaN) 0.0 else wN

        return a * (1 - v) * rBar - v * sigma
    }

    /**
     * This function compute the R_t := F_{t-1} * r_t - delta * | F_t - F_{t-1}| with custom parameters.
     *
     * @param returns the array with the computed returns, it's only between two bounds.
     * @param parameters the custom parameters
     * @param weight the weights of the neural net
     * @param sizeWindow the number of considered elements
     *
     * @return a array of double containing the R_t
     */
    private fun getRt(returns: DoubleArray, parameters: Parameters, weight: Weights, sizeWindow: Int): DoubleArray {

        // it's our index to iterate through the array.
        var t = 1
        // the array we will return
        val rt = DoubleArray(returns.size)
        var ft = Array(1,{ Math.signum(0.0)})
        val mutableWeight = weight.copy()

        var pt = Array(1, {1.0})
        // we init the memory of position
        val position = Position(pt.last(), pt.last(), ft.last())

        while (t < returns.size) {

            // we compute the ft
            val computedFt = computeFt(t, mutableWeight, ft.last(), sizeWindow, returns)

            // we put it in the second layer
            ft = ft.plus(computeRiskAndPerformance(computedFt, parameters, position))

            // store the result
            rt[t] =  ft[ft.lastIndex - 1] * returns[t] - delta * Math.abs(ft[t] - ft[t - 1])

            // update the pt
            pt = pt.plus(pt.last() + rt[t])
            // add the current pnl
            position.currentPnl = pt.last()

            // the loop increment
            t++
        }

        return rt
    }

    /**
     * Given a certain field and a standard deviation, we compute a new parameters objects. We change only the
     * given field.
     *
     * @param field the name of the field we want to change.
     *
     * @return a new parameters object with a new value
     */
    private fun generateNewParameters(field: String ): Parameters {

        val returnedParameters = Parameters(this.delta, this.eta, this.rho, this.x, this.y)

        when (field) {
            "x" -> returnedParameters.x = centredNormalRandom(this.x, this.x/10 + epsilon)
            "y" -> returnedParameters.y = centredNormalRandom(this.y, this.y/10 + epsilon)
        }
        return returnedParameters
    }

    /**
     * We get a sample with a certain mean and a certain standard deviation.
     *
     * @param mean the centred value
     * @param std the standard deviation
     *
     * @return a sample.
     */
    private fun centredNormalRandom(mean: Double, std: Double): Double {
        return NormalDistribution(mean, std).sample()
    }

    /**
     * This function use random walk to optimize the parameters values. The parallel version.
     *
     * @param a is a fixed value cf. the article
     * @param v is a fixed value cf. the article
     * @param returns is the array of returns
     * @param weight the weight of the neural net
     * @param sizeWindow the numbers of considered elements
     *
     * @return an optimized parameters
     */
    fun parallelUpdateParameters(a: Double, v: Double, returns: DoubleArray, weight: Weights,
                         sizeWindow: Int): Parameters {

        // we compute the current value of our parameters : it will be the first "best" result
        var result = this.costFunction(a, v, returns, weight, sizeWindow)
        best = Pair(result, this)
        val executor = Executors.newFixedThreadPool(maxOf(Runtime.getRuntime().availableProcessors(),4))

        // for every field
        for (field in arrayListOf("x", "y")) {
            // run 15 times (15 is a fixed value in the article
            for (notUsed in 0 until 15) {

                val worker = Runnable {
                    // -the generation
                    val newParameters = best.second.generateNewParameters(field)

                    // -the cost function
                    result = newParameters.costFunction(a, v, returns, weight, sizeWindow)

                    // -compare to the "best" and maybe update it
                    @Synchronized if (result > best.first) best = Pair(result, newParameters)
                }
                executor.execute(worker)
            }
        }
        executor.shutdown()
        while (!executor.isTerminated) { }
        // return the best
        return best.second
    }

    override fun toString(): String {
        return "Parameters(delta=$delta, eta=$eta, rho=$rho, x=$x, y=$y)"
    }
}