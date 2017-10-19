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
    val usefulWeights: DoubleArray = weight.coefficients.sliceArray(0..givenT)
    val usefulReturns: DoubleArray = returns.plus(DoubleArray(sizeWindow))
            .sliceArray(maxOf(0,givenT - sizeWindow + 3)..givenT)
            .reversedArray()

    // we zip the two array together and do the multiplication/sum
    for ((first,second) in usefulWeights.zip(usefulReturns)) {
        sum += first * second
    }

    return sum
}

/**
 * This is the implementation of layer 2.
 * We use the threshold and the stop trailing loss to choose if we validate the signal or not.
 * Position 0.0 means we do nothing, so we keep the position
 *
 * @param computedFt the computed signal
 * @param oldFt the previous validate signal
 * @param parameters the object containing x and y
 * @param position the object with best price of the position and the previous position
 */
internal fun computeRiskAndPerformance(
        computedFt : Double, oldFt: Double, parameters: Parameters, position: Position) : Double {

    val res : Double
    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(computedFt) < parameters.y) {
        return position.lastPosition
    } else {
        res = Math.signum(computedFt)
    }

    // if the sign are the same we need to check the loss
    if (res == oldFt) {
        // we compute the diff between price
        val diff = position.currentPrice - position.lastPositionPrice
        // if the trend is rising, it means this diff will be positive, currentPrice > lastPositionPrice
        // if the trend is falling, it means this diff will be negative, currentPrice > lastPositionPrice
        // we need to check we are in the right direction
        if ( res * diff > 0.0 ){
            // it means we have the same sign as the trend
            position.lastPositionPrice = position.currentPrice
            position.lastPosition = res
            position.holdPosition = true
            return res
        } else {
            // we need to check the loss. If big than our rate, we close the position
            if (Math.abs(diff) > parameters.x * 0.001) {
                position.holdPosition = false
                position.lastPosition = res * -1.0
                return res * -1.0
            }
        }
    } else {
        // if the sign are different, we update the lastPositionProfit
        position.lastPositionPrice = position.currentPrice
        position.holdPosition = true
    }

    return res
}