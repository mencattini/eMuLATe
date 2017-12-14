package machine.learning

import java.util.*

/**
 * The weights class. This is the weights for the neural net with the associated methods.
 *
 * @author Romain Mencattini
 *
 * @param sizeWindow : the size of the weights vector
 * @param index : the current iteration
 */
internal class Weights(private val sizeWindow : Int, private val index: Int) {

    var coefficients : DoubleArray
    private var oldDiffFt : DoubleArray
    var oldAt : Double
    var oldBt : Double
    private var magnitude : DoubleArray
    private var rho : DoubleArray

    init {
        // we init the value with random
        val random = Random()

        // create the oldAt and oldBt
        // the default value are different to avoid the division by 0 in weight update
        oldAt = 0.0
        oldBt = 0.0

        // create an array of weight with size of $sizeWindow
        // the weight is defined : (w_{0,M}, vThreshold, w_{M+1})
        coefficients = DoubleArray(sizeWindow, {random.nextDouble()})
        coefficients[coefficients.lastIndex - 1] = 0.0
        // we need to store the diffFt value for the next update
        oldDiffFt = kotlin.DoubleArray(sizeWindow)
        magnitude = kotlin.DoubleArray(sizeWindow, {1.0})
        rho = DoubleArray(sizeWindow, {0.01})
    }

    /**
     * Some kind of constructor. Build a Weights object with the coefficients.
     *
     * @param coefficients an array of double. This will become our coefficients.
     * @param index some kind of timestampe
     * @param at the old A(t)
     * @param bt the old B(t)
     *
     * @return a new Weight object with the given coefficient.
     */
    private fun Weights(coefficients : DoubleArray, index : Int, at : Double, bt: Double) : Weights {
        val weights = Weights(coefficients.size, index)
        weights.coefficients = coefficients
        oldAt = at
        oldBt = bt
        return weights
    }

    /**
     * Make the computation to update the coefficients. According to theory it's a gradient ascent.
     *
     * @param rt is the returns at time t
     * @param ft is equal to F(t)
     * @param ftMinusOne is equal to F(t-1)
     * @param param a parameters object which contains the used values
     * @param returns is the array of previous returns.
     *
     * @return a new Weights object with the new coefficients.
     */
    fun updateWeights(rt: Double, ftMinusOne: Double, ft: Double,
                      param : Parameters, returns: DoubleArray) {

        val oldAt = this.oldAt
        val oldBt = this.oldBt
        val oldDiffFt = this.oldDiffFt.clone()

        val diffRt: Double
        val diffRtMinusOne: Double

        // we compute At, Bt, deltaAt and deltaBt
        val deltaAt = (rt - oldAt)
        val deltaBt = (rt * rt - oldBt)
        val at = oldAt + param.eta * deltaAt
        val bt = oldBt + param.eta * deltaBt

        val wMplusOne = this.wMplusOne()

        if (ft == ftMinusOne) {
            diffRt = 0.0
            diffRtMinusOne = rt
        } else {
            val div = (ft - ftMinusOne) / (Math.abs(ft - ftMinusOne))
            // the dR_{t} / dF_{t}
            diffRt = -param.delta * div
            // the dR_{t} / F_{t-1}
            diffRtMinusOne = rt + param.delta * div
        }

        // we start the computation of dF_t / dw_{i,t}
        // we need to modify the returns before, so we create a new variable
        // we compute the dF_{t} / dw_{i,t} (where it's partial derivation)
        val tmpReturns = returns.plus(ftMinusOne).reversed().toDoubleArray()

        val diffFt = DoubleArray(oldDiffFt.size)

        // according to article, the derivation is dDt / dRt = (B_{t-1} - A_{t-1} * R_t) / (B_{t-1} - A_{t-1}^2)^3/2
        var diffDt = (oldBt - oldAt * rt) / Math.pow(Math.abs(oldBt - oldAt * oldAt), 3 / 2.0)
        if (diffDt.isNaN()) diffDt = -100.0

        // w_{i,t-1} + rho * (diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne))
        val newCoefficients = DoubleArray(this.coefficients.size)
        for (i in this.coefficients.indices) {
            // dF_t / dw_{i,t}
            diffFt[i] = tmpReturns.getDefault(i, 0.0) + oldDiffFt[i] * wMplusOne

            val tmp = (diffDt * (diffRt * diffFt[i] + diffRtMinusOne * oldDiffFt[i]))
            magnitude[i] += Math.pow(tmp, 2.0)
            newCoefficients[i] = this.coefficients[i] + rho[i] * tmp
        }

        this.coefficients = newCoefficients.clone()
        this.oldAt = at
        this.oldBt = bt
        // we store the current diffFt as oldDiffFt for the next iteration
        this.oldDiffFt = diffFt.clone()
        this.rho = rho.zip(magnitude).map { (first, second) -> first / second }.toDoubleArray()
    }

    /**
     * This is an accessor to the private field.
     * @return the w_{M+1} weights
     */
    fun wMplusOne() : Double {
        return coefficients.last()
    }

    /**
     * This is an accessor to the private field.
     * @return the v_t weight
     */
    fun vThreshold() : Double {
        return coefficients[coefficients.lastIndex - 1]
    }

    override fun toString(): String {
        return "Weights(" +
                "coefficients=${Arrays.toString(coefficients.sliceArray(0..coefficients.lastIndex - 2))}," +
                "\nvThreshold=${vThreshold()}," +
                "\nw_{M+1}=${wMplusOne()})"
    }

    /**
     * This function create a copy of the weight object.
     *
     * @param coefficients the (w_i, vthreshold, w_{M+1})
     * @param index the current iteration
     * @param oldAt the old result of at
     * @param oldBt the old result of bt
     * @param oldDiffFt the old result of ft derivation
     *
     * @return a new copied object weight
     */
    fun copy(index: Int = this.index, coefficients: DoubleArray = this.coefficients,
             oldDiffFt : DoubleArray = this.oldDiffFt,oldAt : Double = this.oldAt,
             oldBt: Double = this.oldBt): Weights {
        val res = this.Weights(coefficients, index, oldAt, oldBt)
        res.oldDiffFt = oldDiffFt
        return res
    }

}

private fun DoubleArray.getDefault(index: Int, d: Double = 0.0): Double {
    return if (index < this.size) {
        this[index]
    } else {
        d
    }
}
