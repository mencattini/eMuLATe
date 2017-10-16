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
    private var oldAt : Double
    private var oldBt : Double

    init {
        // we init the value with random
        val random = Random()

        // create the oldAt and oldBt
        // the default value are different to avoid the division by 0 in weight update
        oldAt = 1.0
        oldBt = 0.0

        // create an array of weight with size of $sizeWindow
        // the weight is defined : (w_{0,M}, vThreshold, w_{M+1})
        coefficients = DoubleArray(sizeWindow + 1, {random.nextDouble()})
        // we need to store the diffFt value for the next update
        oldDiffFt = kotlin.DoubleArray(sizeWindow + 1)
    }

    /**
     * Some kind of constructor. Build a Weights object with the coefficients.
     *
     * @param coefficients an array of double. This will become our coefficients.
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
     * @param givenT is the current time.
     * @param param a parameters object which contains the used values
     * @param returns is the array of previous returns.
     *
     * @return a new Weights object with the new coefficients.
     */
    fun updateWeights(rt: Double, ftMinusOne: Double, ft: Double,
                      givenT: Int, param : Parameters, returns: DoubleArray): Weights {

        val diffRt : Double
        val diffRtMinusOne: Double
        if (ft == ftMinusOne) {
            diffRt = 1.0
            diffRtMinusOne = rt
        } else {
            // the dR_{t} / dF_{t}
            diffRt = ((-param.delta * (ft - ftMinusOne))
                    / (Math.abs(ft - ftMinusOne)))
            // the dR_{t} / F_{t-1}
            diffRtMinusOne = rt + ((param.delta * (ft - ftMinusOne))
                    / (Math.abs(ft - ftMinusOne)))
        }

        // we start the computation of dF_t / dw_{i,t}
        // we need to multiple dF_{t-1} / dw_{i,t-1}) * (\delta F_t / \delta F_{t-1})
        var diffFtMinusOneBis = oldDiffFt.map { it -> it * this.wMplusOne() }

        // we need to modify the returns before, so we create a new variable
        var tmpReturns = DoubleArray(sizeWindow + 1)
        if (returns.size > sizeWindow ) {
            // we have to many returns, we need to slice to get the right number.
            // reverse the array
            tmpReturns = returns.reversed()
                    // take to sizeWindow - 1 (NOT INCLUDED)
                    .slice(0..sizeWindow)
                    // adding the last element
                    .plus(ftMinusOne)
                    // the cast
                    .toDoubleArray()
        } else {
            // we need to add enough 0 to returns to avoid reduction of diffFtMinusOneBis with map
            for ((i, ele) in returns.reversed().toDoubleArray().plus(ftMinusOne).withIndex()) {
                tmpReturns[i] = ele
            }
        }

        // dF_t / dw_{i,t}
        var diffFt = tmpReturns.zip(diffFtMinusOneBis)
                .map { (first, second) -> first + second }

        // we need to do : dF_t / dw_{i,t} * dR_t / dF_t
        diffFt = diffFt.map { it -> it * diffRt }

        // dR_t / dF_{t-1} * dF_{t-1} / dw_{i,t-1}
        diffFtMinusOneBis = oldDiffFt.map { it -> it * diffRtMinusOne }


        // we compute At, Bt, deltaAt and deltaBt
        val deltaAt = (rt - oldAt)
        val deltaBt = (rt * rt - oldBt)
        val at = oldAt + param.eta * deltaAt
        val bt = oldBt + param.eta * deltaBt

        // according to article, the derivation is dDt / dRt = (B_{t-1} - A_{t-1} * R_t) / (B_{t-1} - A_{t-1}^2)^3/2
        val diffDt = (oldBt - oldAt * rt) / Math.pow(Math.abs(oldBt - oldAt * oldAt), 3/2.0)

        // diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne)
        val deltaW = diffFt.zip(diffFtMinusOneBis)
                .map { (first, second) -> (first + second) * diffDt}.toDoubleArray()

        // the updating delta using weights = weights + rho * deltaW
        val res = Weights(coefficients.zip(deltaW)
                .map { (first, second) -> first + param.rho * second }
                .toDoubleArray(), givenT + 1, at, bt)
        // we store the current diffFt as oldDiffFt for the next iteration
        res.oldDiffFt = diffFt.toDoubleArray()
        return res
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