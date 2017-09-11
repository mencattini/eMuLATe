package machine.learning

import java.util.*

/**
 * The weights class. This is the weights for the neural net with the associated methods.
 *
 * @author Romain mencattini
 */
class Weights(private val sizeWindow : Int, val index: Int) {

    private var coefficients : DoubleArray

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
    private fun updateWeights(givenT: Int, param : Parameters, ft: DoubleArray, returns: DoubleArray): Weights {
        // first we need to compute the delta w_{i,t}

        // the diff(R_{t}, F_{t})
        val diffRt = ((param.delta * (ft[givenT - 1] - ft[givenT]))
                / (Math.abs(ft[givenT] - ft[givenT - 1])))
        // the diff(R_{t}, F_{t-1})
        val diffRtMinusOne = ((param.delta * (ft[givenT] - ft[givenT - 1]))
                / (Math.abs(ft[givenT] - ft[givenT - 1])) + returns[givenT])

        // diff(F_T, w_{i,t}) and every elements is multiplied by diffRt
        val diffFt = returns.reversed().toDoubleArray().plus(ft[givenT - 1])
                .map { it -> it * diffRt }

        // diff(F_{T-1}, w_{i, t - 1} and every elements is multiplied by diffRtMinusOne
        val diffFtMinusOne = returns.reversed().toDoubleArray().plus(ft[givenT - 2])
                .map {it -> it * diffRtMinusOne}

        // i don't know the derivation of Dt by Rt, so i use a constant until i find it.
        val diffDt = 1

        // diffDt * (diffRt * diffFt + diffRtMinusOne * diffFtMinusOne)
        val deltaW = diffFt.zip(diffFtMinusOne)
                .map { (first, second) -> (first + second) * diffDt}.toDoubleArray()

        // the updating delta using weights = weights + rho * deltaW
         return Weights(coefficients.zip(deltaW)
                .map { (first, second) -> first + param.rho * second }
                 .toDoubleArray(), givenT + 1)

    }
}