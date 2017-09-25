package machine.learning

import java.util.*

class Parameters() {

    var delta : Double
    var eta : Double
    var rho : Double
    var x : Double
    var y : Double

    init {
        // we init the value with random
        val random = Random()

        delta = random.nextDouble()
        eta = random.nextDouble()
        rho = random.nextDouble()
        x = random.nextDouble()
        y = random.nextDouble()
    }

    fun Parameters(delta: Double, eta: Double, rho: Double, x : Double, y : Double): Parameters {

        val param = Parameters()
        param.delta = delta
        param.eta = eta
        param.rho = rho
        param.x = x
        param.y = y

        return param
    }

    override fun toString(): String {
        return "Parameters(delta=$delta, eta=$eta, rho=$rho, x=$x, y=$y)"
    }


}