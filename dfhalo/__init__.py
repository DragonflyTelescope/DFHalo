__version__ = "0.1"

# Pixel scale (arcsec/pixel) for reduced and raw Dragonfly data
DF_pixel_scale = 2.5
DF_raw_pixel_scale = 2.85

# Gain (e-/ADU) of Dragonfly
DF_Gain = 0.37
    
from . import utils
from . import profile

from . import clustering
from . import pipe

from . import plot
from . import atlas

