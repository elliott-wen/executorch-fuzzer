# Confirmed ENN operator bugs — index

All device-verified on Exynos E9965 (Solomon), 5/5 deterministic. Run any repro with the broker up
(job-port 15554) and the E9965 worker connected: `python <repro>.py`.

| repro | mechanism | operators |
|---|---|---|
| [repro_int_datamovement.py](repro_int_datamovement.py) | WRONG-VALUE (garbage read) + ZEROED (dropped tail store) | `permute_copy`, `roll`, `t_copy`, `transpose_copy.int`, `select_copy.int`, `constant_pad_nd`, `expand_copy`, `squeeze_copy.dim/.dims`, `select_scatter` |
| [repro_div_scalar_overflow.py](repro_div_scalar_overflow.py) | NONFINITE (fp16 overflow, finite input) | `div.Scalar` |
| [repro_zeroed.py](repro_zeroed.py) | ZEROED (dropped store) | `embedding`, `mul.Scalar`, `pixel_unshuffle` |

Additional confirmed operators (see per-op table in `../README.md`; not given a dedicated repro —
same replay mechanism applies via `_replay.py <job_id>`): `rsub.Scalar`, `sub.out`, `add.Scalar`,
`glu.out`, `maximum.out`, `avg_pool2d.out`, `pixel_shuffle`, `hardtanh`, `bmm.out`, `view_copy`,
`unsqueeze_copy`, `prod.int_out`/`sum.IntList_out` (uint8 saturation, flagged), `remainder.Tensor_out`
(SCALE:-0.5).
