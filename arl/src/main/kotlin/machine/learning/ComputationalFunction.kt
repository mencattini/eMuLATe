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

    var res : Double
    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(sum) < parameters.y) {
        return Math.signum(0.0)
    } else {
        res = Math.signum(sum)
    }

    // if the sign are the same we need to check the loss
    if (Math.signum(res) == Math.signum(oldFt)) {
        // we compute the diff between profit
        val diff = positionProfit.lastPositionProfit - positionProfit.currentProfit
        // if the diff is negative, it means the lastPositionProfit is less than currentProfit
        // so we update the position
        if (diff < 0.0 ){
            positionProfit.lastPositionProfit = positionProfit.currentProfit
            return res
        } else {
            // we need to check the loss. If big than our rate, we close the position
            if (diff < parameters.x * 0.001) {
                return Math.signum(0.0)
            }
        }
    } else {
        // if the sign are different, we update the lastPositionProfit
        positionProfit.lastPositionProfit = positionProfit.currentProfit
    }
    return res
}