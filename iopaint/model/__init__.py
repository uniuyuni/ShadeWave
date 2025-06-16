
from .fcf import FcF
from .lama import LaMa
from .ldm import LDM
from .mat import MAT
from .mi_gan import MIGAN
from .zits import ZITS

models = {
    LaMa.name: LaMa,
    LDM.name: LDM,
    ZITS.name: ZITS,
    MAT.name: MAT,
    FcF.name: FcF,
    MIGAN.name: MIGAN,
}
