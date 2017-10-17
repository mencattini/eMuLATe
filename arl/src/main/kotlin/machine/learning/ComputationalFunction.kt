package machine.learning

/**
 * Compute the Ft layer using weights, vthreshold, returns and old_ft.
 * Given t, my need are :
 * - compute ft
 * - return the value and the sign
 *
 * @param givenT an Int. It's our index.
 * @param weight the weights of the neural net
 * @param oldFt the pair where first = the sign, second = the value, resulting of F_t
 * @param sizeWindow the number of considered items
 * @param returns the array of computed returns
 * @param parameters the parameters
 *
 * @return a pair of signum and value
 */
internal fun computeFt(givenT: Int, weight: Weights, oldFt: Double, sizeWindow: Int,
                       returns: DoubleArray, parameters: Parameters, positionProfit: PositionProfit): Double {

    // this part doesn't depends on index
    var sum = weight.wMplusOne() * oldFt + weight.vThreshold()


    // we get the useful weights and returns
    // we add some 0 to avoid the out of bound array exception.
    val usefulWeights: DoubleArray = weight.coefficients.sliceArray(0..givenT)
    val usefulReturns: DoubleArray = returns.plus(DoubleArray(sizeWindow))
            .sliceArray(maxOf(0,givenT - sizeWindow + 3)..givenT)
            .reversedArray()

    // we zip the two array together and do the multiplication/sum
    for ((first,second) in usefulWeights.zip(usefulReturns)) {
        sum += first * second
    }

    var res = Math.signum(0.0)
    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(sum) < parameters.y) {
        return Math.signum(0.0)
    } else {
        res = Math.signum(sum)
    }

    // stop trailing loss
    if (Math.signum(res) == Math.signum(oldFt) && res != Math.signum(0.0)) {
        // we need to check the loss if we keep the same position
        if (positionProfit.lastPositionProfit - positionProfit.currentProfit < parameters.x * 0.001){
            res = Math.signum(0.0)
        }
    } else if (Math.signum(res) != Math.signum(oldFt)) {
        positionProfit.lastPositionProfit = positionProfit.currentProfit
    }
    return res
}