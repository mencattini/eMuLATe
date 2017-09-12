package machine.learning

import java.util.*

class Parameters() {

    public var delta : Double
    public var eta : Double
    public var rho : Double
    public var x : Double
    public var y : Double

    init {
        // we init the value with random
        val random = Random()

        delta = random.nextDouble()
        eta = random.nextDouble()
        rho = random.nextDouble()
        x = random.nextDouble()
        y = random.nextDouble()
    }

    public fun Parameters(delta: Double, eta: Double, rho: Double, x : Double, y : Double) {

        var param = Parameters()
        param.delta = delta
        param.eta = eta
        param.rho = rho
        param.x = x
        param.y = y
    }

    override fun toString(): String {
        return "Parameters(delta=$delta, eta=$eta, rho=$rho, x=$x, y=$y)"
    }


}