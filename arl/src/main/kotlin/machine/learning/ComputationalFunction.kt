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
                       returns: DoubleArray, parameters: Parameters, positionPrice: PositionPrice): Double {

    // this part doesn't depends on index
    var sum = weight.wMplusOne() * oldFt + weight.vThreshold()


    // we get the useful weights and returns
    // we add some 0 to avoid the out of bound array execption.
    val usefulWeights: DoubleArray = weight.coefficients.sliceArray(0..givenT)
    val usefulReturns: DoubleArray = returns.plus(DoubleArray(sizeWindow))
            .sliceArray(maxOf(0,givenT - sizeWindow + 3)..givenT)
            .reversedArray()

    // we zip the two array together and do the multiplication/sum
    for ((first,second) in usefulWeights.zip(usefulReturns)) {
        sum += first * second
    }


    val neutral = Math.signum(0.0)
    val res :Double

    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(sum) < parameters.y) {
        return neutral
    } else {
        res = Math.signum(sum)
    }

    // if the ft doesn't change and it's not a 0.0, we need to check the loss
    if (oldFt == res) {
        // the difference between the currentPrice and the lastPositionPrice.
        // if the different is negative, it means the trend goes down, if it's positive, the trend goes up
        val diff = positionPrice.currentPrice - positionPrice.lastPositionPrice
        // if it goes up, the good answer is +1, so positive times positive, => positive
        // if it goes down, the good answer is -1, so negative, times negative => positive
        // it means, if diff times position, i.e. {-1,+1} is negative, we do a bad choice and we need to controlate
        // our loss.
        if (diff * res < parameters.x * -0.001) {
            // we change the signal, to get the opposite
            return neutral
        }
    } else if (oldFt != res){
        // if oldFt and res are different, and not neutral, it means we update the last position price
        positionPrice.lastPositionPrice = positionPrice.currentPrice
        return res
    }

    // then we return the res
    return res
}