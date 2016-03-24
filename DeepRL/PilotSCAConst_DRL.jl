module PilotSCAConst_DRL

export RangeMax, RangeMin, ThetaMin, ThetaMax, PsiMin, PsiMax, SpeedOwnMin, SpeedOwnMax
export SpeedIntMin, SpeedIntMax
export ThetaDim, PsiDim, SpeedOwnDim, SpeedIntDim
export Thetas, Psis, SpeedOwns, SpeedInts, Actions


const RangeMax = 3000.0 #m
const RangeMin = 0.0 #ft
const ThetaMin = -pi #[rad]
const ThetaMax = pi  #[rad]
const PsiMin   = -pi #[rad]
const PsiMax   = pi  #[rad]

const SpeedOwnMin = 10.0 #10.0  # [m/s]
const SpeedOwnMax = 20.0 #20.0  # [m/s]
const SpeedIntMin = 10.0
const SpeedIntMax = 20.0

const PsiDim = 41#37 #37 #3
const ThetaDim = 41
const SpeedOwnDim = 11 #5 #2
const SpeedIntDim = 11



const Thetas          = linspace(ThetaMin,ThetaMax,ThetaDim)
const Psis            = linspace(PsiMin,PsiMax,PsiDim)
const SpeedOwns       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
const SpeedInts       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

const Actions = [-20.0,-10.0,0.0,10.0,20.0,-6.0]

end # module PilotSCAConst_DRL