#!/usr/bin/env python3
"""CONFIRMED (38/38, 5x) Neutron bug: int8 relu returns 0 at the positive-saturation boundary.
When the quantized input reaches the top int8 code (dequantizes to +127.0), eager relu(127)=127
but Neutron returns 0.0 — consistent with the +127 code overflowing to -128 (sign bit set), which
relu then clamps to the zero-point. Mechanism: WRONG-VALUE / ZEROED at the int8 max (delegated,
ops=3: quantize -> Neutron relu -> dequantize). abs delegates cleanly; only relu hits this.
Replays exact corpus .pte on the eIQ NSYS simulator."""
from _replay import replay
replay(["w0:424","w1:466","w109:469","w109:241"])
