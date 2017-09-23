package machine.learning

import java.util.*

/**
 * The weights class. This is the weights for the neural net with the associated methods.
 *
 * @author Romain mencattini
 */
class Weights(private val sizeWindow : Int, val index: Int) {

    var coefficients : DoubleArray
    var oldDiffFt : DoubleArray
    private var oldAt : Double
    private var oldBt : Double

    init {
        // we init the value with random
        val random = Random()

        // create the oldAt and oldBt
        // the default value are different to avoid the division by 0 in weigth update
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
     * @param givenT is the current time.
     * @param param a parameters object which contains the used values
     * @param ft is the array of previous ft.
     * @param returns is the array of previous returns.
     *
     * @return a new Weights object with the new coefficients.
     */
    fun updateWeights(givenT: Int, param : Parameters, ft: Array<Pair<Double, Double>>, returns: DoubleArray): Weights {

        val rt = returns[givenT - 1]
        // the diff(R_{t}, F_{t})
        var diffRt = (( -param.delta * (ft[givenT].first - ft[givenT - 1].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first) + 0.01 ))
        diffRt = if (diffRt == 0.0) 1.0 else diffRt

        // the diff(R_{t}, F_{t-1})
        var diffRtMinusOne =  rt  + ((param.delta * (ft[givenT].first - ft[givenT - 1].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first) + 0.01 ) + rt)
        diffRtMinusOne = if (diffRtMinusOne == 0.0) 1.0 else diffRtMinusOne

        // we need to multiple diff(F_{t-1},w_{i,t-1}) by diff(F_t, F_{t-1})
        var diffFtMinusOneBis = oldDiffFt.map { it -> it * coefficients.last() }

        // derivation(F_t, w_{i,t}) = diff(F_t, w_{i,t}) + diff(F_t, F_{t-1}) * diff(F_{t-1},w_{i,t-1})
        // we need to modify the returns before, so we create a new variable
        var tmpReturns = DoubleArray(sizeWindow + 1)
        if (returns.size > sizeWindow ) {
            // we have to many returns, we need to slice to get the right number.
            // reverse the array
            tmpReturns = returns.reversed()
                    // take to sizeWindow - 1 (NOT INCLUDED)
                    .slice(0..(sizeWindow ))
                    // adding the last element
                    .plus(ft[givenT - 1].first)
                    // the cast
                    .toDoubleArray()
        } else {
            // we need to add enough 0 to returns to avoid reduction of diffFtMinusOneBis with map
            for ((i, ele) in returns.reversed().toDoubleArray().plus(ft[givenT - 1].first).withIndex()) {
                tmpReturns[i] = ele
            }
        }
        var diffFt = tmpReturns.zip(diffFtMinusOneBis)
                .map { it -> it.first + it.second }

        // we need to multiply derivation(F_t, w_{i,t}) with diffRt
        diffFt = diffFt.map { it -> it * diffRt }

        // we need to multiply derivation(F_{t-1}, w_{i,t-1}) with diffRtMinusOne
        diffFtMinusOneBis = oldDiffFt.map { it -> it * diffRtMinusOne }


        // we compte At, Bt, deltaAt and deltBt
        val deltaAt = (rt - oldAt)
        val deltaBt = (rt * rt - oldBt)
        val at = oldAt + param.eta * deltaAt
        val bt = oldBt + param.eta * deltaBt


        // accorind to article, the derivation is dDt / dRt = (B_{t-1} - A_{t-1} * R_t) / (B_{t-1} - A_{t-1}^2)^3/2
        val diffDt = (oldBt - oldAt * rt) / Math.pow(Math.abs(oldBt - oldAt * oldAt), 3/2.0)

        // diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne)
        val deltaW = diffFt.zip(diffFtMinusOneBis)
                .map { (first, second) -> (first + second) * diffDt}.toDoubleArray()

        // the updating delta using weights = weights + rho * deltaW
        var res = Weights(coefficients.zip(deltaW)
                .map { (first, second) -> first + param.rho * second }
                .toDoubleArray(), givenT + 1, at, bt)
        // we store the current diffFt as oldDiffFt for the next iteration
        res.oldDiffFt = diffFt.toDoubleArray()
        return res
    }

    fun wMplusOne() : Double {
        return coefficients.last()
    }

    fun vThreshold() : Double {
        return coefficients[coefficients.lastIndex - 1]
    }

    override fun toString(): String {
        return "Weights(" +
                "coefficients=${Arrays.toString(coefficients.sliceArray(0..coefficients.lastIndex - 2))}," +
                "\nvThreshold=${vThreshold()}," +
                "\nw_{M+1}=${wMplusOne()})"
    }


}