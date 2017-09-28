package machine.learning

/**
 * Compute the Ft layer using weights, vthreshold, returns and old_ft.
 * Given t, my need are :
 * - compute ft
 * - return the value and the sign
 *
 * @param givenT an Int. It's our index.
 * @param weight the weights of the neural net
 * @param ft the pair where first = the sign, second = the value, resulting of F_t
 * @param sizeWindow the number of considered items
 * @param returns the array of computed returns
 * @param parameters the parameters
 *
 * @return a pair of signum and value
 */
internal fun computeFt(givenT: Int, weight: Weights, ft:Array<Pair<Double,Double>>,
                       sizeWindow: Int, returns: DoubleArray, parameters: Parameters): Pair<Double, Double> {

    // this part doesn't depends on index
    var sum = weight.wMplusOne() * ft[givenT - 1].first + weight.vThreshold()


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
        // using the maxof(0, t-sizedwindow + 2) to avoid negative index
        // reverse the array for the same thing than above
        usefulWeights = weight.coefficients.sliceArray(0..(sizeWindow - 2))
        usefulReturns = returns.sliceArray(maxOf(0, givenT - sizeWindow + 2)..givenT).reversedArray()
    }

    // we zip the two array together and do the multiplication/sum
    for ((wi, ri) in usefulWeights.zip(usefulReturns)) {
        sum += wi * ri
    }

    // TODO: adding the x check not sure how.

    // we check the threshold
    // if it's greater than the threshold, we keep the result
    if (Math.abs(sum) > parameters.y) {
        return Pair(Math.signum(sum), sum)
    }
    // else we just do nothing
    return Pair(Math.signum(0.0), 0.0)
}