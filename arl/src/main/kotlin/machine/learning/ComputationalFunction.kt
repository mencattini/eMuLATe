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
 *
 * @return a pure signal
 */
internal fun computeFt(givenT: Int, weight: Weights, oldFt: Double, sizeWindow: Int, returns: DoubleArray): Double {

    // this part doesn't depends on index
    var sum = weight.wMplusOne() * oldFt + weight.vThreshold()


    // we get the useful weights and returns
    // we add some 0 to avoid the out of bound array exception.
    val usefulWeights: DoubleArray = weight.coefficients.sliceArray(0..minOf(givenT, sizeWindow - 3))
    val usefulReturns: DoubleArray = returns.plus(DoubleArray(sizeWindow))
            .sliceArray(maxOf(0,givenT - sizeWindow + 3)..givenT)
            .reversedArray()

    // we zip the two array together and do the multiplication/sum
    for (i in 0..usefulWeights.lastIndex){
        sum += usefulReturns[i] * usefulWeights[i]
    }

    return sum
}

/**
 * This is the implementation of layer 2.
 * We use the threshold and the stop trailing loss to choose if we validate the signal or not.
 * Position 0.0 means we do nothing, so we keep the position
 *
 * @param computedFt the computed signal
 * @param parameters the object containing x and y
 * @param position the object with best price of the position and the previous position
 *
 * @return the right signal according to the risk and performance layer.
 */
internal fun computeRiskAndPerformance(
        computedFt : Double, parameters: Parameters, position: Position) : Double {

    var res: Double
    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(computedFt) < parameters.y) {
        return Math.signum(0.0)
    } else {
        res = Math.signum(computedFt)
    }

    // if we change the position we need to update the max pnl and the position
    if (res != position.lastPosition) {
        position.maxPnl = position.currentPnl
        position.lastPosition = res
        return res
    } else {
        // else we need to check the diff
        val diff = position.maxPnl - position.currentPnl
        if (diff <= 0.0) {
            position.maxPnl = position.currentPnl
        } else if (diff > 0.0 && diff > parameters.x) {
            res = Math.signum(0.0)
        }
    }
    return res
}