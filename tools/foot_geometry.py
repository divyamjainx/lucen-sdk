# -*- coding: utf-8 -*-
"""tools/foot_geometry.py - collision box of the biped_Laika foot, in the foot (ankle_roll link) frame, in metres.

Used by the simulators: `--foot-box` replaces the foot collision meshes with this box, and the sole corners are
taken from it. Each entry is (min, max) along x (forward), y (left), z (up). The frame origin is near the front
of the sole, not at the ankle pitch axis.
"""

FOOT_BOX = ((-0.125, 0.051), (-0.040, 0.040), (-0.059, 0.030))
