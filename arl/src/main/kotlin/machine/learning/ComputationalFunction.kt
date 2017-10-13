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
    val usefulWeights: DoubleArray
    val usefulReturns: DoubleArray

    // if the t is smaller than our windows, we just take the t first elements
    // we sub 2 because the last weight is used with the F_{t-1}
    if (givenT < sizeWindow - 2) {
        // as said in the formal neural net layer :
        // w_{i,t} * r_{t-i}
        // so we need to reverse the returns array
        usefulWeights = weight.coefficients.sliceArray(0..givenT)
        usefulReturns = returns.sliceArray(0..givenT).reversedArray()

    } else {
        // we sub the (sizeWindow - 2) to always get the same number of elements than the weights
        // reverse the array for the same thing than above
        usefulWeights = weight.coefficients.sliceArray(0..(sizeWindow - 3))
        usefulReturns = returns.sliceArray( (givenT - sizeWindow + 3)..givenT).reversedArray()
    }

    // we zip the two array together and do the multiplication/sum
    sum += usefulWeights.zip(usefulReturns)
            .map { (first, second) -> first * second }
            .reduce{total, next -> total + next}


    val neutral = Math.signum(0.0)
    var res = neutral

    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(sum) > parameters.y) {
        res = Math.signum(sum)
    }

    // TODO: fixe it, because it doesn't work
    // if the ft doesn't change and it's not a 0.0, we need to check the loss
    if (oldFt == res && res != neutral) {
        // the difference between the currentPrice and the lastPositionPrice.
        // if the different is negative, it means the trend goes down, if it's positive, the trend goes up
        val diff = positionPrice.currentPrice - positionPrice.lastPositionPrice
        // if it goes up, the good answer is +1, so positive times positive, => positive
        // if it goes down, the good answer is -1, so negative, times negative => positive
        // it means, if diff times position, i.e. {-1,+1} is negative, we do a bad choice and we need to controlate
        // our loss.
        if (diff * res < 0.0 && Math.abs(diff) > parameters.x * 0.0001) {
            // we change the signal, to get the opposite
            res = neutral
        }
    } else if (oldFt != res && res != neutral){
        // if oldFt and res are different, and not neutral, it means we update the last position price
        positionPrice.lastPositionPrice = positionPrice.currentPrice
    }

    // then we return the res
    return res
}