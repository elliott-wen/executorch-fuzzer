| operator | total | deleg | MIS-fin | MIS-nf | SKIP | verdict | mechanism |
|---|--:|--:|--:|--:|--:|---|---|
| `rsub.Scalar` | 439 | 439 | 358 | 36 | 5 | CONFIRMED | WRONG-VALUE |
| `gelu.out` | 440 | 440 | 4 | 373 | 0 | CONFIRMED | WRONG-VALUE |
| `sub.Scalar` | 403 | 403 | 229 | 125 | 1 | CONFIRMED | ZEROED |
| `add.Scalar` | 391 | 391 | 208 | 120 | 2 | CONFIRMED | WRONG-VALUE |
| `_softmax.out` | 444 | 444 | 6 | 281 | 0 | CONFIRMED | ZEROED |
| `div.Scalar` | 305 | 305 | 1 | 283 | 0 | CONFIRMED | NONFINITE |
| `mul.Scalar` | 204 | 204 | 94 | 99 | 0 | CONFIRMED | ZEROED |
| `sub.out` | 379 | 379 | 158 | 23 | 46 | CONFIRMED | WRONG-VALUE |
| `clamp.out` | 372 | 372 | 21 | 108 | 165 | CONFIRMED | WRONG-VALUE |
| `select_scatter` | 386 | 386 | 107 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `hardtanh` | 416 | 416 | 104 | 0 | 125 | CONFIRMED | WRONG-VALUE |
| `logit` | 386 | 386 | 23 | 79 | 0 | CONFIRMED | WRONG-VALUE |
| `select_copy.int` | 378 | 378 | 91 | 0 | 4 | CONFIRMED | SCALE:-2.184e+04 |
| `maximum.out` | 359 | 359 | 85 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `pixel_shuffle` | 598 | 598 | 82 | 1 | 327 | CONFIRMED | WRONG-VALUE |
| `permute_copy` | 379 | 379 | 54 | 1 | 2 | CONFIRMED | WRONG-VALUE |
| `leaky_relu.out` | 444 | 444 | 6 | 37 | 0 | INTERMITTENT |  |
| `sqrt.out` | 442 | 442 | 0 | 43 | 64 | CONFIRMED | WRONG-VALUE |
| `mul.out` | 271 | 271 | 1 | 39 | 0 | INTERMITTENT |  |
| `rsqrt.out` | 303 | 303 | 0 | 39 | 2 | CONFIRMED | WRONG-VALUE |
| `pixel_unshuffle` | 728 | 728 | 34 | 0 | 147 | CONFIRMED | ZEROED |
| `glu.out` | 437 | 437 | 28 | 2 | 1 | CONFIRMED | WRONG-VALUE |
| `bmm.out` | 321 | 321 | 2 | 23 | 0 | CONFIRMED | WRONG-VALUE |
| `sum.IntList_out` | 407 | 146 | 23 | 0 | 14 | CONFIRMED | WRONG-VALUE |
| `transpose_copy.int` | 432 | 88 | 20 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `hardtanh.out` | 435 | 435 | 19 | 0 | 237 | INTERMITTENT |  |
| `remainder.Tensor_out` | 433 | 113 | 18 | 0 | 37 | CONFIRMED | SCALE:-0.5 |
| `prod.int_out` | 391 | 150 | 17 | 0 | 48 | CONFIRMED | WRONG-VALUE |
| `expand_copy` | 347 | 347 | 17 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `squeeze_copy.dims` | 389 | 389 | 14 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `logit.out` | 398 | 398 | 0 | 13 | 97 | INTERMITTENT |  |
| `bitwise_left_shift.Tensor_out` | 441 | 281 | 2 | 9 | 47 | CONFIRMED | WRONG-VALUE |
| `relu` | 414 | 414 | 7 | 2 | 79 | CONFIRMED | WRONG-VALUE |
| `squeeze_copy.dim` | 388 | 388 | 9 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `avg_pool2d.out` | 10 | 10 | 9 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `t_copy` | 433 | 75 | 7 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `_log_softmax.out` | 417 | 417 | 3 | 4 | 0 | INTERMITTENT |  |
| `view_copy` | 361 | 361 | 6 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `div.out` | 437 | 437 | 3 | 3 | 20 | INTERMITTENT |  |
| `var.correction_out` | 438 | 271 | 0 | 6 | 0 | CONFIRMED | WRONG-VALUE |
| `linear.out` | 14 | 14 | 0 | 6 | 0 | CONFIRMED | WRONG-VALUE |
| `bitwise_xor.Scalar_out` | 441 | 299 | 4 | 0 | 97 | INTERMITTENT |  |
| `minimum.out` | 352 | 352 | 2 | 0 | 76 | INTERMITTENT |  |
| `bitwise_and.Scalar_out` | 441 | 266 | 2 | 0 | 91 | INTERMITTENT |  |
| `bitwise_or.Scalar_out` | 438 | 297 | 2 | 0 | 105 | INTERMITTENT |  |
| `roll` | 392 | 387 | 1 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `embedding` | 6 | 6 | 1 | 0 | 5 | CONFIRMED | ZEROED |
| `linear` | 13 | 13 | 0 | 1 | 0 | CONFIRMED | WRONG-VALUE |
| `constant_pad_nd` | 22 | 22 | 1 | 0 | 0 | CONFIRMED | WRONG-VALUE |
| `unsqueeze_copy` | 395 | 395 | 1 | 0 | 1 | CONFIRMED | WRONG-VALUE |
| `bitwise_right_shift.Tensor_out` | 440 | 293 | 0 | 0 | 55 | not-gated |  |
| `bitwise_or.Tensor_out` | 436 | 280 | 0 | 0 | 107 | not-gated |  |
| `fmod.Tensor_out` | 431 | 101 | 0 | 0 | 33 | not-gated |  |
| `div.out_mode` | 424 | 34 | 0 | 0 | 12 | not-gated |  |
| `replication_pad1d.out` | 438 | 438 | 0 | 0 | 438 | not-gated |  |
| `min.unary_out` | 441 | 88 | 0 | 0 | 7 | not-gated |  |
| `floor_divide.out` | 438 | 135 | 0 | 0 | 78 | not-gated |  |
| `replication_pad3d.out` | 433 | 433 | 0 | 0 | 433 | not-gated |  |
| `bitwise_xor.Tensor_out` | 437 | 259 | 0 | 0 | 110 | not-gated |  |
| `bitwise_and.Tensor_out` | 443 | 282 | 0 | 0 | 111 | not-gated |  |
| `max.unary_out` | 442 | 96 | 0 | 0 | 3 | not-gated |  |
| `floor_divide` | 410 | 72 | 0 | 0 | 30 | not-gated |  |
| `replication_pad2d.out` | 436 | 436 | 0 | 0 | 436 | not-gated |  |
| `remainder.Scalar_out` | 432 | 45 | 0 | 0 | 32 | not-gated |  |
| `stack` | 67 | 27 | 0 | 0 | 14 | not-gated |  |
| `pow.Scalar_out` | 436 | 28 | 0 | 0 | 17 | not-gated |  |

<!-- SUMMARY
distinct ops in corpus: 199
delegated-failing ops: 66
confirmed-bug ops: 40
total delegated MIS-fin: 1884  MIS-nf: 1756  SKIP: 3761
portable(ruled-out) MIS+CRASH: 1299
-->
