package machine.learning

import java.util.*

/**
 * The weights class. This is the weights for the neural net with the associated methods.
 *
 * @author Romain mencattini
 */
class Weights(private val sizeWindow : Int, val index: Int) {

    public var coefficients : DoubleArray
    public var oldDiffFt : DoubleArray

    init {
        // we init the value with random
        val random = Random()

        // create an array of weight with size of $sizeWindow
        coefficients = DoubleArray(sizeWindow, {random.nextDouble()})
        // we need to store the diffFt value for the next update
        oldDiffFt = kotlin.DoubleArray(sizeWindow)
    }

    /**
     * Some kind of constructor. Build a Weights object with the coefficients.
     *
     * @param coefficients an array of double. This will become our coefficients.
     *
     * @return a new Weight object with the given coefficient.
     */
    public fun Weights(coefficients : DoubleArray, index : Int) : Weights {
        var weights = Weights(coefficients.size, index)
        weights.coefficients = coefficients
        return weights
    }

    /**
     * Make the computation to update the coefficients. According to theory it's a gradient ascent.
     *
     * @param givenT is the current time.
     * @param delta is the trading cost.
     * @param rho is the learning rate.
     * @param ft is the array of previous ft.
     * @param returns is the array of previous returns.
     *
     * @return a new Weights object with the new coefficients.
     */
    public fun updateWeights(givenT: Int, param : Parameters, ft: Array<Pair<Double, Double>>, returns: DoubleArray): Weights {
        // first we need to compute the delta w_{i,t}

        // the diff(R_{t}, F_{t})
        val diffRt = ((param.delta * (ft[givenT - 1].first - ft[givenT].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first) + 0.01 ))
        // the diff(R_{t}, F_{t-1})
        val diffRtMinusOne = ((param.delta * (ft[givenT].first - ft[givenT - 1].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first) + 0.01 ) + returns[givenT - 1])

        // we need to multiple diff(F_{t-1},w_{i,t-1}) by diff(F_t, F_{t-1})
        var diffFtMinusOneBis = oldDiffFt.map { it -> it * coefficients.last() }

        // derivation(F_t, w_{i,t}) = diff(F_t, w_{i,t}) + diff(F_t, F_{t-1}) * diff(F_{t-1},w_{i,t-1})
        // we need to modify the returns before, so we create a new variable
        var tmpReturns = DoubleArray(sizeWindow)
        if (returns.size > sizeWindow - 1) {
            // we have to many returns, we need to slice to get the right number.
            // reverse the array
            tmpReturns = returns.reversed()
                    // take to sizeWindow - 1 (NOT INCLUDED)
                    .slice(0..(sizeWindow - 1))
                    // adding the last element
                    .plus(ft[givenT - 1].first)
                    // the cast
                    .toDoubleArray()
        } else {
            // we need to add enough 0 to returns to avoid reduction of diffFtMinusOneBis with map
            var i = 0
            for (ele in returns.reversed().toDoubleArray().plus(ft[givenT - 1].first)) {
                tmpReturns[i] = ele
                i++
            }
        }
        var diffFt = tmpReturns.zip(diffFtMinusOneBis)
                .map { it -> it.first + it.second }

        // we need to multiply derivation(F_t, w_{i,t}) with diffRt
        diffFt = diffFt.map { it -> it * diffRt }

        // we need to multiply derivation(F_{t-1}, w_{i,t-1}) with diffRtMinusOne
        diffFtMinusOneBis = oldDiffFt.map { it -> it * diffRtMinusOne }

        // i don't know the derivation of Dt by Rt, so i use a constant until i find it.
        val diffDt = 1

        // diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne)
        val deltaW = diffFt.zip(diffFtMinusOneBis)
                .map { (first, second) -> (first + second) * diffDt}.toDoubleArray()

        // the updating delta using weights = weights + rho * deltaW
        var res = Weights(coefficients.zip(deltaW)
                .map { (first, second) -> first + param.rho * second }
                .toDoubleArray(), givenT + 1)
        // we store the current diffFt as oldDiffFt for the next iteration
        res.oldDiffFt = diffFt.toDoubleArray()
        return res
    }
}