# Moto G54 — corrected attribution

corpus=corpus_v2/vulkan  graphs=76801  (OK/SKIP/MISMATCH/CRASH per skip-log)

## Per-output MISMATCH (co-occurrence confound removed)

Baseline per-output mismatch rate = 6.98%. 'real-bug rate' excludes non-finite (upstream nan/inf propagation).

| op | trials | fails | rate | real-bug rate | numeric | dtype | shape | nonfinite | old whole-graph blame |
|----|-------:|------:|-----:|--------------:|--------:|------:|------:|----------:|----------------------:|
| `clamp.Tensor_out` | 928 | 576 | 62.1% | **55.5%** | 514 | 1 | 0 | 61 | 1360 |
| `select_scatter` | 886 | 463 | 52.3% | **51.4%** | 26 | 429 | 0 | 8 | 925 |
| `index_put` | 284 | 154 | 54.2% | **49.6%** | 141 | 0 | 0 | 13 | 552 |
| `bitwise_left_shift.Tensor_out` | 281 | 66 | 23.5% | **23.5%** | 66 | 0 | 0 | 0 | 305 |
| `topk.values` | 312 | 77 | 24.7% | **21.8%** | 54 | 0 | 14 | 9 | 250 |
| `sum.IntList_out` | 899 | 195 | 21.7% | **21.6%** | 194 | 0 | 0 | 1 | 633 |
| `bitwise_left_shift.Tensor_Scalar` | 388 | 65 | 16.8% | **16.8%** | 65 | 0 | 0 | 0 | 346 |
| `diagonal_copy` | 327 | 58 | 17.7% | **16.5%** | 54 | 0 | 0 | 4 | 320 |
| `gather.out` | 559 | 88 | 15.7% | **12.7%** | 71 | 0 | 0 | 17 | 300 |
| `tril.out` | 276 | 40 | 14.5% | **12.7%** | 32 | 3 | 0 | 5 | 269 |
| `any.all_out` | 982 | 116 | 11.8% | **11.8%** | 115 | 1 | 0 | 0 | 549 |
| `bitwise_left_shift.Tensor_Scalar_out` | 397 | 45 | 11.3% | **11.3%** | 45 | 0 | 0 | 0 | 247 |
| `remainder.Scalar` | 982 | 123 | 12.5% | **11.3%** | 107 | 4 | 0 | 12 | 646 |
| `floor_divide.out` | 683 | 100 | 14.6% | **10.4%** | 71 | 0 | 0 | 29 | 472 |
| `pixel_unshuffle` | 550 | 62 | 11.3% | **10.4%** | 55 | 2 | 0 | 5 | 342 |
| `remainder.Scalar_out` | 963 | 98 | 10.2% | **9.7%** | 93 | 0 | 0 | 5 | 556 |
| `any.dims_out` | 1923 | 181 | 9.4% | **9.4%** | 178 | 3 | 0 | 0 | 1037 |
| `prod.int_out` | 949 | 86 | 9.1% | **9.0%** | 85 | 0 | 0 | 1 | 514 |
| `floor_divide` | 783 | 105 | 13.4% | **8.7%** | 68 | 0 | 0 | 37 | 528 |
| `pow.Tensor_Tensor_out` | 825 | 261 | 31.6% | **7.8%** | 64 | 0 | 0 | 197 | 784 |
| `expand_copy` | 859 | 81 | 9.4% | **7.7%** | 65 | 1 | 0 | 15 | 586 |
| `bitwise_right_shift.Tensor_Scalar` | 397 | 30 | 7.6% | **7.6%** | 28 | 2 | 0 | 0 | 219 |
| `sub.out` | 837 | 74 | 8.8% | **7.3%** | 60 | 1 | 0 | 13 | 505 |
| `copy` | 603 | 46 | 7.6% | **6.8%** | 41 | 0 | 0 | 5 | 330 |
| `slice_scatter` | 891 | 67 | 7.5% | **6.5%** | 57 | 1 | 0 | 9 | 440 |
| `clone` | 840 | 72 | 8.6% | **6.4%** | 53 | 1 | 0 | 18 | 549 |
| `bitwise_right_shift.Tensor_Scalar_out` | 427 | 27 | 6.3% | **6.3%** | 27 | 0 | 0 | 0 | 211 |
| `clamp.out` | 956 | 146 | 15.3% | **6.3%** | 59 | 1 | 0 | 86 | 629 |
| `remainder.Tensor_out` | 934 | 85 | 9.1% | **6.2%** | 58 | 0 | 0 | 27 | 531 |
| `linear` | 533 | 45 | 8.4% | **6.2%** | 33 | 0 | 0 | 12 | 255 |
| `transpose_copy.int` | 918 | 65 | 7.1% | **6.0%** | 55 | 0 | 0 | 10 | 548 |
| `permute_copy` | 1088 | 77 | 7.1% | **6.0%** | 64 | 1 | 0 | 12 | 565 |
| `select_copy.int` | 747 | 54 | 7.2% | **5.9%** | 41 | 3 | 0 | 10 | 403 |
| `gt.Tensor_out` | 845 | 49 | 5.8% | **5.8%** | 49 | 0 | 0 | 0 | 386 |
| `flip` | 1007 | 70 | 7.0% | **5.8%** | 55 | 3 | 0 | 12 | 553 |
| `view_copy` | 1075 | 80 | 7.4% | **5.7%** | 60 | 1 | 0 | 19 | 586 |
| `cumsum.out` | 832 | 50 | 6.0% | **5.6%** | 47 | 0 | 0 | 3 | 464 |
| `where.self_out` | 200 | 12 | 6.0% | **5.5%** | 10 | 1 | 0 | 1 | 133 |
| `linear.out` | 551 | 57 | 10.3% | **5.4%** | 30 | 0 | 0 | 27 | 278 |
| `abs.out` | 930 | 60 | 6.5% | **5.4%** | 48 | 2 | 0 | 10 | 482 |

## Whole-graph CRASH enrichment (per-output can't apply to a process abort)

Baseline crash-rate among ran graphs = 9.6%.

| op | support | crashes | crash-rate | lift | oddsR |
|----|--------:|--------:|-----------:|-----:|------:|
| `unfold_copy` | 2092 | 844 | 40.3% | 4.21 | 8.3 |
| `narrow_copy` | 3238 | 1172 | 36.2% | 3.78 | 7.9 |
| `scatter.src_out` | 278 | 99 | 35.6% | 3.72 | 5.4 |
| `scatter.value_out` | 218 | 68 | 31.2% | 3.26 | 4.4 |
| `native_layer_norm` | 248 | 76 | 30.6% | 3.20 | 4.3 |
| `scatter_add.out` | 281 | 79 | 28.1% | 2.94 | 3.8 |
| `clone` | 1738 | 412 | 23.7% | 2.48 | 3.2 |
| `lift_fresh_copy` | 1885 | 418 | 22.2% | 2.32 | 2.9 |
| `index_put` | 380 | 79 | 20.8% | 2.17 | 2.5 |
| `alias_copy` | 1839 | 378 | 20.6% | 2.15 | 2.6 |
| `div.out_mode` | 281 | 57 | 20.3% | 2.12 | 2.4 |
| `split_with_sizes_copy` | 1125 | 227 | 20.2% | 2.11 | 2.5 |
| `clamp.Tensor_out` | 902 | 175 | 19.4% | 2.03 | 2.3 |
| `replication_pad3d.out` | 371 | 70 | 18.9% | 1.97 | 2.2 |
| `tril.out` | 687 | 113 | 16.4% | 1.72 | 1.9 |
| `transpose_copy.int` | 1812 | 296 | 16.3% | 1.71 | 1.9 |
| `expand_copy` | 1570 | 239 | 15.2% | 1.59 | 1.8 |
| `copy` | 1127 | 165 | 14.6% | 1.53 | 1.7 |
| `max.dim_max` | 260 | 36 | 13.8% | 1.45 | 1.5 |
| `min.dim_min` | 267 | 36 | 13.5% | 1.41 | 1.5 |
| `pow.Tensor_Tensor_out` | 1076 | 145 | 13.5% | 1.41 | 1.5 |
| `clampr` | 2427 | 323 | 13.3% | 1.39 | 1.5 |
| `pad` | 1320 | 172 | 13.0% | 1.36 | 1.4 |
| `pixel_unshuffle` | 1062 | 134 | 12.6% | 1.32 | 1.4 |
| `bitwise_left_shift.Tensor_out` | 533 | 67 | 12.6% | 1.31 | 1.4 |
| `where.self` | 684 | 81 | 11.8% | 1.24 | 1.3 |
| `sum.IntList_out` | 1473 | 168 | 11.4% | 1.19 | 1.2 |
| `select_scatter` | 1049 | 117 | 11.2% | 1.17 | 1.2 |
| `rshp` | 1544 | 167 | 10.8% | 1.13 | 1.2 |
| `bitwise_left_shift.Tensor_Scalar` | 744 | 79 | 10.6% | 1.11 | 1.1 |
