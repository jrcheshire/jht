"""Shared probe grid for the high-l recursion error-growth measurement.

Pure Python (no deps) so it imports in *both* the project env (jht side,
``highL_recursion_growth.py``) and the ephemeral mpmath env (oracle side,
``highL_mpmath_ref.py`` run via ``pixi exec --spec mpmath --spec numpy``).
Keep the two halves on an identical grid by importing this from each.
"""

from __future__ import annotations

import math

# Colatitudes (radians), kept off the exact poles where the sectoral seed
# underflows to 0 -- the near-pole corner is already gated in test_recursion.
THETA = [0.3, 0.9, 1.5, 2.2, 2.8]

# Orders: low m (most l-terms accumulate in the contraction), a mid spread,
# and high m (the sin(theta)**m sectoral-underflow regime).  A probe at order m
# is only evaluated for l >= m.
M_LIST = [0, 1, 2, 50, 200, 800, 1500]

# Band limits to report the error at.  500 is the *anchor*: jht is known correct
# to ~1e-12 vs scipy there, so ref-vs-jht agreement at 500 pins the convention
# before trusting the high-l rows.  The rest probe FAR past any usable HEALPix
# geometry (1.5*nside=2048 -> ceiling 3072) to map the fp64 roundoff-growth law of
# the recursion ITSELF -- "how high can the core numerics go" decoupled from nside.
L_LIST = [500, 2000, 8000, 16000, 32000]
LMAX = max(L_LIST)

SPINS = [0, 2]

DPS = 50  # mpmath working precision for the reference (>> fp64; isolates roundoff)

# natural amplitude of a normalized (spin-weighted) lambda column: the
# addition-theorem bound |lambda^s_{l,m}| <= sqrt((2l+1)/4pi).  Root-insensitive
# scale for the relative error (avoids the meaningless blow-up at a zero crossing).
def amp_scale(ell: int) -> float:
    return math.sqrt((2.0 * ell + 1.0) / (4.0 * math.pi))
