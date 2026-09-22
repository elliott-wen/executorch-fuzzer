# Confirmed Neutron bugs — index

| repro | operator | mechanism |
|---|---|---|
| [repro_relu_int8_saturation.py](repro_relu_int8_saturation.py) | `relu` | int8 relu returns 0 at the +127 saturation boundary (delegated, 38/38 confirmed 5×) |

Run with the nxp broker (job-port 15574) + an `nxp_client` worker up: `python repro_relu_int8_saturation.py`.
