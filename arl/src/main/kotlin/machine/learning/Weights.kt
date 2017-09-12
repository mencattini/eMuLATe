package machine.learning

import java.util.*

/**
 * The weights class. This is the weights for the neural net with the associated methods.
 *
 * @author Romain mencattini
 */
class Weights(private val sizeWindow : Int, val index: Int) {

    public var coefficients : DoubleArray

    init {
        // we init the value with random
        val random = Random()

        // create an array of weight with size of $sizeWindow
        coefficients = DoubleArray(sizeWindow, {random.nextDouble()})
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
    private fun updateWeights(givenT: Int, param : Parameters, ft: Array<Pair<Double, Double>>, returns: DoubleArray,
                              diffFtMinusOne: List<Double>): Weights {
        // first we need to compute the delta w_{i,t}

        // the diff(R_{t}, F_{t})
        val diffRt = ((param.delta * (ft[givenT - 1].first - ft[givenT].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first)))
        // the diff(R_{t}, F_{t-1})
        val diffRtMinusOne = ((param.delta * (ft[givenT].first - ft[givenT - 1].first))
                / (Math.abs(ft[givenT].first - ft[givenT - 1].first)) + returns[givenT])

        // we need to multiple diff(F_{t-1},w_{i,t-1}) by diff(F_t, F_{t-1})
        var diffFtMinusOneBis = diffFtMinusOne.map { it -> it * coefficients.last() }

        // derivation(F_t, w_{i,t}) = diff(F_t, w_{i,t}) + diff(F_t, F_{t-1}) * diff(F_{t-1},w_{i,t-1})
        var diffFt = returns.reversed().toDoubleArray().plus(ft[givenT - 1].first)
                .zip(diffFtMinusOneBis)
                .map { it -> it.first + it.second }

        // we need to multiply derivation(F_t, w_{i,t}) with diffRt
        diffFt = diffFt.map { it -> it * diffRt }

        // we need to multiply derivation(F_{t-1}, w_{i,t-1}) with diffRtMinusOne
        diffFtMinusOneBis = diffFtMinusOne.map { it -> it * diffRtMinusOne }

        // i don't know the derivation of Dt by Rt, so i use a constant until i find it.
        val diffDt = 1

        // diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne)
        val deltaW = diffFt.zip(diffFtMinusOneBis)
                .map { (first, second) -> (first + second) * diffDt}.toDoubleArray()

        // the updating delta using weights = weights + rho * deltaW
         return Weights(coefficients.zip(deltaW)
                .map { (first, second) -> first + param.rho * second }
                 .toDoubleArray(), givenT + 1)

    }
}