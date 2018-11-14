package it.unibo.casestudy

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._

class Gradient extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with FieldUtils with ExplicitFields {

  override type MainResult = Any
  override def main = {
    rep((0,0.0)){ case (k,r) =>
      val nr = if(k<30) nextRandom() else r
      node.put("r", nr)
      (k+1, nr)
    }
    gradientField(mid==0)
  }

  def gradientField(src: Boolean) = rep(Double.PositiveInfinity)(distance =>
    mux(src){ 0.0 }{
      (fnbr(distance)+fsns(nbrRange)).minHoodPlus
    }
  )
}