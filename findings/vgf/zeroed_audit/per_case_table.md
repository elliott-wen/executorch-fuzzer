| job      | out | verdict | target_op               | trigger_ops                  | deleg | A(3)  | B(3)  | B_plan    | comp_ancestor_probe      |
|----------|-----|---------|-------------------------|------------------------------|-------|-------|-------|-----------|--------------------------|
| w101:421 | 3   | COMP    | fmod                    | sum                          | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n3             |
| w102:77  | 0   | COMP    | pixel_shuffle           | fill                         | 6     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w110:633 | 1   | COMP    | abs                     | fill                         | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w118:68  | 2   | COMP    | detach_copy             | div;pow                      | 4     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n4,n1          |
| w11:260  | 0   | COMP    | bitwise_right_shift     | _to_copy;floor_divide;neg;sl | 6     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n5,n1,n2,n6,n4 |
| w120:152 | 0   | COMP    | sqrt                    | fill;roll                    | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w121:583 | 1   | COMP    | asinh                   | bitwise_left_shift;sigmoid   | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w12:424  | 0   | COMP    | slice_scatter           | _to_copy;mean;broadcast_to;f | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n6,n5,n7,n1    |
| w14:27   | 1   | COMP    | remainder               | floor_divide                 | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n3             |
| w15:430  | 3   | COMP    | clone                   | floor_divide                 | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n7             |
| w18:752  | 2   | COMP    | neg                     | fill                         | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n3             |
| w19:347  | 0   | COMP    | split_copy              | fill                         | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w19:548  | 0   | COMP    | prod                    | pow;prod                     | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n1,n2          |
| w19:81   | 0   | COMP    | clamp                   | fill                         | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2             |
| w2:119   | 1   | COMP    | bitwise_or              | max;slice_scatter            | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n1,n0          |
| w33:310  | 1   | COMP    | scatter                 | _to_copy;pixel_shuffle;slice | 5     | O/O/O | Z/Z/Z | multiOut  | anc_ok                   |
| w36:147  | 0   | COMP    | max                     | fill;mul                     | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w37:33   | 0   | COMP    | neg                     | trunc;_native_batch_norm_leg | 1     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n1,n0          |
| w37:604  | 0   | COMP    | repeat                  | bitwise_right_shift;round;wh | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n1,n0       |
| w3:174   | 0   | COMP    | expm1                   | copy                         | 2     | O/O/O | Z/Z/Z | 1outDeleg | anc_untestable           |
| w40:2    | 0   | COMP    | var_mean                | max;sin;broadcast_to;expand_ | 6     | O/O/O | Z/Z/Z | multiOut  | ANC_WRONG:n8             |
| w42:33   | 0   | COMP    | unfold_copy             | fill                         | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w43:562  | 1   | COMP    | prod                    | slice_scatter                | 1     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w49:644  | 0   | COMP    | logical_and             | remainder;fmod;argmin        | 0     | O/O/O | Z/Z/Z | noDeleg   | ANC_WRONG:n0,n1,n2       |
| w54:229  | 1   | COMP    | flip                    | fill;pixel_shuffle           | 1     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w56:121  | 2   | COMP    | logical_xor             | _to_copy;slice_copy;_to_copy | 4     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n10,n11,n5,n6  |
| w57:613  | 0   | COMP    | where                   | broadcast_to;fill;_to_copy   | 1     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n0,n1       |
| w60:389  | 1   | COMP    | le                      | view_copy;cos;_to_copy;bitwi | 0     | O/O/O | Z/Z/Z | noDeleg   | ANC_WRONG:n4,n1,n3,n0    |
| w61:468  | 1   | COMP    | gt                      | constant_pad_nd;_to_copy;cla | 3     | O/O/O | Z/Z/Z | multiOut  | ANC_WRONG:n6             |
| w61:595  | 0   | COMP    | gather                  | _to_copy;slice_copy;clamp;sq | 3     | O/O/O | Z/Z/Z | multiOut  | anc_ok                   |
| w62:643  | 0   | COMP    | sigmoid                 | bitwise_left_shift           | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w64:737  | 0   | COMP    | trunc                   | fill                         | 1     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w65:767  | 0   | COMP    | abs                     | trunc;fill                   | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n1          |
| w67:428  | 1   | COMP    | ne                      | slice_scatter;le             | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w68:60   | 1   | COMP    | mul                     | prod;fill                    | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n1          |
| w74:450  | 3   | COMP    | abs                     | fmod                         | 0     | O/O/O | Z/Z/Z | noDeleg   | ANC_WRONG:n3             |
| w76:777  | 1   | COMP    | minimum                 | view_copy;unbind_copy;ne     | 5     | O/O/O | Z/Z/Z | multiOut  | anc_ok                   |
| w78:168  | 1   | COMP    | remainder               | clone;floor_divide           | 4     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n1          |
| w84:104  | 0   | COMP    | index_select            | clamp;lift_fresh_copy;slice_ | 3     | O/O/O | Z/Z/Z | multiOut  | anc_ok                   |
| w84:353  | 0   | COMP    | unfold_copy             | bitwise_left_shift;exp       | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0,n1          |
| w86:679  | 0   | COMP    | pow                     | fill                         | 3     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w87:250  | 1   | COMP    | roll                    | view_copy;fill;sigmoid       | 4     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n2,n0,n1       |
| w88:13   | 0   | COMP    | maximum                 | min;squeeze_copy;fill        | 4     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w93:501  | 0   | COMP    | argmin                  | remainder                    | 0     | O/O/O | Z/Z/Z | noDeleg   | ANC_WRONG:n1             |
| w97:48   | 0   | COMP    | gt                      | native_group_norm            | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n0             |
| w9:354   | 2   | COMP    | sign                    | _native_batch_norm_legit;_to | 2     | O/O/O | Z/Z/Z | 1outDeleg | ANC_WRONG:n1             |
| w101:730 | 1   | SIBL    | tanh                    | where                        | 15    | O/O/O | Z/Z/Z | multiOut  |                          |
| w103:40  | 0   | SIBL    | _upsample_bilinear2d_aa | scatter_add                  | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w108:171 | 0   | SIBL    | fmod                    | split_with_sizes_copy        | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w109:130 | 0   | SIBL    | constant_pad_nd         | detach_copy                  | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w109:539 | 0   | SIBL    | argmax                  | permute_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w10:13   | 1   | SIBL    | squeeze_copy            | constant_pad_nd              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w111:578 | 0   | SIBL    | min                     | log10                        | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w113:429 | 1   | SIBL    | erf                     | squeeze_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w113:546 | 0   | SIBL    | any                     | logical_or                   | 2     | O/O/O | Z/Z/Z | 1outDeleg |                          |
| w113:641 | 3   | SIBL    | isnan                   | amax                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w113:751 | 1   | SIBL    | round                   | min                          | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w115:354 | 0   | SIBL    | clamp                   | abs                          | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w115:540 | 1   | SIBL    | trunc                   | clone                        | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w115:631 | 0   | SIBL    | prod                    | floor;var_mean               | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w117:156 | 1   | SIBL    | max                     | alias_copy                   | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w117:643 | 0   | SIBL    | squeeze_copy            | squeeze_copy                 | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w118:120 | 2   | SIBL    | max                     | transpose_copy               | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w11:274  | 2   | SIBL    | div                     | logical_xor                  | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w120:186 | 0   | SIBL    | index_select            | split_with_sizes_copy        | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w120:331 | 3   | SIBL    | fmod                    | transpose_copy               | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w121:710 | 0   | SIBL    | neg                     | remainder;fmod               | 10    | O/O/O | Z/Z/Z | multiOut  |                          |
| w121:756 | 1   | SIBL    | flip                    | pixel_unshuffle              | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w123:22  | 1   | SIBL    | div                     | expand_copy                  | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w123:49  | 1   | SIBL    | max                     | mul                          | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w123:59  | 3   | SIBL    | mul                     | clone                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w124:703 | 2   | SIBL    | min                     | squeeze_copy                 | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w125:527 | 0   | SIBL    | logical_and             | detach_copy                  | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w126:77  | 2   | SIBL    | fmod                    | hardtanh                     | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w127:226 | 0   | SIBL    | repeat                  | min                          | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w127:739 | 1   | SIBL    | split_copy              | constant_pad_nd              | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w13:119  | 0   | SIBL    | convolution             | min                          | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w13:22   | 0   | SIBL    | prod                    | fmod                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w13:404  | 0   | SIBL    | sinh                    | floor_divide                 | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w13:423  | 0   | SIBL    | div                     | logical_xor                  | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w14:533  | 1   | SIBL    | max                     | narrow_copy                  | 3     | O/O/O | Z/Z/Z | ?         |                          |
| w15:167  | 0   | SIBL    | permute_copy            | permute_copy                 | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w15:780  | 0   | SIBL    | select_copy             | elu;roll                     | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w16:27   | 1   | SIBL    | mul                     | roll                         | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w16:633  | 0   | SIBL    | lift_fresh_copy         | isnan                        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w17:556  | 1   | SIBL    | masked_scatter          | transpose_copy               | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w17:600  | 0   | SIBL    | pow                     | sinh                         | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w18:124  | 0   | SIBL    | div                     | _native_batch_norm_legit     | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w18:185  | 0   | SIBL    | prod                    | lift_fresh_copy              | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w18:438  | 3   | SIBL    | narrow_copy             | scatter                      | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w19:198  | 0   | SIBL    | log1p                   | acosh                        | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w19:517  | 3   | SIBL    | gelu                    | min                          | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w19:680  | 0   | SIBL    | sigmoid                 | sigmoid                      | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w19:708  | 4   | SIBL    | trunc                   | permute_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w1:287   | 0   | SIBL    | t_copy                  | squeeze_copy                 | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w1:476   | 0   | SIBL    | elu                     | log10                        | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w1:489   | 1   | SIBL    | trunc                   | lift_fresh_copy              | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w20:142  | 2   | SIBL    | split_copy              | split_with_sizes_copy        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w20:350  | 1   | SIBL    | transpose_copy          | transpose_copy               | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w20:762  | 0   | SIBL    | mul                     | sub                          | 14    | O/O/O | Z/Z/Z | multiOut  |                          |
| w21:490  | 1   | SIBL    | prod                    | narrow_copy                  | 2     | O/O/O | Z/Z/Z | ?         |                          |
| w22:623  | 0   | SIBL    | slice_copy              | permute_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w23:651  | 0   | SIBL    | fmod                    | lift_fresh_copy              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w24:150  | 3   | SIBL    | argmax                  | prod                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w24:604  | 0   | SIBL    | div                     | index                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w26:495  | 1   | SIBL    | flip                    | detach_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w27:193  | 3   | SIBL    | maximum                 | mean                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w27:293  | 0   | SIBL    | fmod                    | permute_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w27:445  | 2   | SIBL    | argmax                  | transpose_copy               | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w27:460  | 0   | SIBL    | expm1                   | unsqueeze_copy               | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w27:493  | 1   | SIBL    | logical_and             | trunc                        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w28:227  | 2   | SIBL    | split_copy              | split_copy                   | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w28:338  | 1   | SIBL    | constant_pad_nd         | trunc                        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w28:40   | 2   | SIBL    | flip                    | ceil                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w2:434   | 2   | SIBL    | min                     | pixel_shuffle                | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w2:750   | 0   | SIBL    | squeeze_copy            | lift_fresh_copy              | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w30:39   | 0   | SIBL    | unbind_copy             | pow                          | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w30:584  | 1   | SIBL    | fmod                    | select_scatter               | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w31:551  | 1   | SIBL    | squeeze_copy            | div                          | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w32:577  | 1   | SIBL    | permute_copy            | sinh                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w33:368  | 0   | SIBL    | prod                    | squeeze_copy                 | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w33:738  | 2   | SIBL    | detach_copy             | alias_copy                   | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w34:126  | 0   | SIBL    | prod                    | lift_fresh_copy              | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w34:155  | 0   | SIBL    | prod                    | div                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w34:571  | 1   | SIBL    | amin                    | clone                        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w35:582  | 0   | SIBL    | cosh                    | cosh                         | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w35:585  | 3   | SIBL    | constant_pad_nd         | prod                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w35:66   | 1   | SIBL    | logical_or              | roll                         | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w35:661  | 3   | SIBL    | prod                    | expand_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w36:316  | 3   | SIBL    | index                   | detach_copy;bitwise_or       | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w36:723  | 0   | SIBL    | bitwise_and             | atanh;log10                  | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w37:113  | 2   | SIBL    | div                     | log10                        | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w37:551  | 2   | SIBL    | fmod                    | slice_scatter                | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w38:438  | 0   | SIBL    | narrow_copy             | abs                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w38:506  | 0   | SIBL    | div                     | prod                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w38:544  | 1   | SIBL    | expand_copy             | copy                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w39:508  | 0   | SIBL    | repeat                  | mul                          | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w3:339   | 0   | SIBL    | max                     | sigmoid                      | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w3:401   | 0   | SIBL    | trunc                   | expand_copy                  | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w3:545   | 3   | SIBL    | max                     | max                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w3:570   | 0   | SIBL    | copy                    | constant_pad_nd              | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w41:359  | 1   | SIBL    | prod                    | constant_pad_nd              | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w43:185  | 0   | SIBL    | amax                    | t_copy                       | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w43:420  | 2   | SIBL    | argmax                  | trunc                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w44:280  | 2   | SIBL    | erf                     | pow                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w44:603  | 2   | SIBL    | min                     | alias_copy                   | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w45:187  | 0   | SIBL    | narrow_copy             | mul                          | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w46:357  | 1   | SIBL    | atan                    | prod;acos                    | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w46:527  | 1   | SIBL    | max                     | abs                          | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w46:683  | 1   | SIBL    | prod                    | mul                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w47:429  | 0   | SIBL    | replication_pad1d       | sqrt                         | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w47:757  | 3   | SIBL    | div                     | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w47:758  | 1   | SIBL    | abs                     | squeeze_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w48:439  | 0   | SIBL    | div                     | fmod                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:131  | 0   | SIBL    | view_copy               | ceil                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:26   | 1   | SIBL    | sum                     | fill                         | 1     | O/O/O | Z/Z/Z | 1outDeleg |                          |
| w49:398  | 0   | SIBL    | scatter                 | prod                         | 10    | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:677  | 3   | SIBL    | sin                     | var_mean                     | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:702  | 1   | SIBL    | narrow_copy             | lift_fresh_copy              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:72   | 0   | SIBL    | pow                     | t_copy                       | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w49:772  | 0   | SIBL    | max                     | squeeze_copy                 | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w4:345   | 0   | SIBL    | abs                     | sin                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w50:476  | 2   | SIBL    | flip                    | t_copy                       | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w51:149  | 1   | SIBL    | mean                    | mean                         | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w51:269  | 3   | SIBL    | constant_pad_nd         | squeeze_copy                 | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w51:43   | 0   | SIBL    | eq                      | le                           | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w51:447  | 0   | SIBL    | min                     | copy                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w51:57   | 0   | SIBL    | scatter_add             | tanh;relu                    | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w52:127  | 0   | SIBL    | lift_fresh_copy         | split_with_sizes_copy        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w52:165  | 0   | SIBL    | mul                     | floor_divide                 | 9     | O/O/O | Z/Z/Z | multiOut  |                          |
| w52:291  | 0   | SIBL    | leaky_relu              | mul                          | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w52:442  | 0   | SIBL    | pow                     | var_mean                     | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w52:559  | 1   | SIBL    | flip                    | view_copy                    | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w53:416  | 3   | SIBL    | max                     | squeeze_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w53:684  | 0   | SIBL    | fmod                    | logical_or                   | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w53:750  | 0   | SIBL    | split_copy              | amin                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w54:221  | 1   | SIBL    | slice_copy              | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w54:294  | 3   | SIBL    | fmod                    | clone                        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w55:183  | 1   | SIBL    | transpose_copy          | split_with_sizes_copy        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w55:603  | 0   | SIBL    | relu                    | relu                         | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w56:535  | 4   | SIBL    | roll                    | constant_pad_nd              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w56:575  | 1   | SIBL    | masked_scatter          | transpose_copy               | 9     | O/O/O | Z/Z/Z | multiOut  |                          |
| w57:301  | 0   | SIBL    | abs                     | clone                        | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w57:332  | 2   | SIBL    | masked_scatter          | alias_copy                   | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w57:87   | 2   | SIBL    | prod                    | permute_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w58:774  | 1   | SIBL    | prod                    | clone                        | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w59:476  | 4   | SIBL    | max                     | clone                        | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w5:141   | 3   | SIBL    | argmax                  | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w5:302   | 1   | SIBL    | roll                    | permute_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w5:687   | 3   | SIBL    | mul                     | div;split_with_sizes_copy    | 10    | O/O/O | Z/Z/Z | multiOut  |                          |
| w60:387  | 0   | SIBL    | gelu                    | unbind_copy                  | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w60:659  | 0   | SIBL    | clone                   | alias_copy                   | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w60:740  | 0   | SIBL    | div                     | div                          | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w61:311  | 2   | SIBL    | sinh                    | pow                          | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w61:49   | 2   | SIBL    | gelu                    | view_copy                    | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w61:512  | 0   | SIBL    | logical_and             | split_with_sizes_copy        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w61:683  | 0   | SIBL    | logical_and             | alias_copy                   | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w61:753  | 0   | SIBL    | fmod                    | argmin                       | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w62:202  | 0   | SIBL    | asinh                   | slice_copy;ge                | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w62:307  | 0   | SIBL    | gt                      | fmod                         | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w62:5    | 2   | SIBL    | prod                    | constant_pad_nd              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w63:424  | 1   | SIBL    | argmin                  | unbind_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w63:527  | 0   | SIBL    | fmod                    | pixel_shuffle                | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w63:541  | 2   | SIBL    | trunc                   | constant_pad_nd              | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:223  | 3   | SIBL    | clamp                   | alias_copy                   | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:248  | 1   | SIBL    | trunc                   | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:291  | 2   | SIBL    | min                     | constant_pad_nd              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:397  | 0   | SIBL    | logical_and             | pow                          | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:581  | 0   | SIBL    | remainder               | leaky_relu                   | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w64:745  | 0   | SIBL    | permute_copy            | constant_pad_nd              | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w65:269  | 1   | SIBL    | detach_copy             | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w65:98   | 3   | SIBL    | prod                    | pixel_unshuffle              | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w66:138  | 1   | SIBL    | fmod                    | lift_fresh_copy              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w66:735  | 1   | SIBL    | prod                    | view_copy                    | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w67:144  | 3   | SIBL    | view_copy               | relu                         | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w67:196  | 1   | SIBL    | min                     | view_copy                    | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w68:67   | 0   | SIBL    | view_copy               | view_copy                    | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w69:418  | 0   | SIBL    | sinh                    | argmax                       | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w69:603  | 3   | SIBL    | t_copy                  | transpose_copy               | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w6:20    | 3   | SIBL    | prod                    | permute_copy                 | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w6:341   | 0   | SIBL    | detach_copy             | remainder                    | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w6:357   | 0   | SIBL    | min                     | alias_copy                   | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w6:46    | 4   | SIBL    | lt                      | expand_copy                  | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w70:133  | 2   | SIBL    | squeeze_copy            | slice_scatter                | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w70:67   | 2   | SIBL    | fmod                    | argmax                       | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w71:21   | 1   | SIBL    | abs                     | abs                          | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w71:48   | 3   | SIBL    | min                     | alias_copy                   | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w71:78   | 2   | SIBL    | abs                     | abs                          | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w72:16   | 1   | SIBL    | ceil                    | maximum                      | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w72:343  | 2   | SIBL    | transpose_copy          | alias_copy                   | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w72:64   | 3   | SIBL    | expm1                   | log10                        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w72:736  | 1   | SIBL    | fmod                    | clone                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w73:314  | 0   | SIBL    | remainder               | view_copy                    | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w73:453  | 3   | SIBL    | max                     | squeeze_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w73:739  | 0   | SIBL    | hardtanh                | abs                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w75:25   | 0   | SIBL    | flip                    | view_copy                    | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w76:576  | 0   | SIBL    | flip                    | fmod                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w76:75   | 3   | SIBL    | flip                    | detach_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w77:21   | 1   | SIBL    | atanh                   | atanh                        | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w77:382  | 1   | SIBL    | logical_or              | squeeze_copy                 | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w77:565  | 3   | SIBL    | div                     | constant_pad_nd              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w78:31   | 1   | SIBL    | clone                   | unbind_copy                  | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w78:58   | 0   | SIBL    | repeat                  | tan                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w78:645  | 2   | SIBL    | maximum                 | maximum                      | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w78:662  | 1   | SIBL    | topk                    | floor_divide                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w79:547  | 2   | SIBL    | permute_copy            | expand_copy                  | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w7:307   | 0   | SIBL    | clone                   | detach_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w7:326   | 0   | SIBL    | squeeze_copy            | remainder                    | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w7:43    | 1   | SIBL    | abs                     | unbind_copy                  | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w81:202  | 0   | SIBL    | pow                     | div                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w81:230  | 2   | SIBL    | min                     | permute_copy                 | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w81:334  | 1   | SIBL    | fmod                    | squeeze_copy                 | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w82:482  | 0   | SIBL    | sin                     | log10                        | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w82:57   | 0   | SIBL    | fmod                    | flip                         | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w83:215  | 0   | SIBL    | cumsum                  | alias_copy                   | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w84:515  | 2   | SIBL    | squeeze_copy            | lift_fresh_copy              | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w85:22   | 3   | SIBL    | tan                     | fill                         | 4     | O/O/O | Z/Z/Z | ?         |                          |
| w85:89   | 0   | SIBL    | max                     | expand_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w86:232  | 3   | SIBL    | replication_pad2d       | transpose_copy               | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w87:107  | 1   | SIBL    | abs                     | var_mean;sub                 | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w87:385  | 1   | SIBL    | remainder               | unbind_copy                  | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w87:488  | 2   | SIBL    | index                   | lift_fresh_copy              | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w88:570  | 0   | SIBL    | constant_pad_nd         | view_copy                    | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w88:642  | 0   | SIBL    | clamp                   | abs                          | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w89:137  | 1   | SIBL    | remainder               | pow                          | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w89:431  | 2   | SIBL    | div                     | split_with_sizes_copy        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w8:94    | 2   | SIBL    | detach_copy             | amin                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w90:95   | 2   | SIBL    | prod                    | squeeze_copy                 | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w93:111  | 1   | SIBL    | unfold_copy             | acosh                        | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w93:40   | 0   | SIBL    | tanh                    | mul                          | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w93:520  | 0   | SIBL    | div                     | masked_fill                  | 2     | O/O/O | Z/Z/Z | ?         |                          |
| w93:72   | 0   | SIBL    | atan2                   | clone                        | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w95:135  | 0   | SIBL    | max                     | clone                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w95:446  | 1   | SIBL    | mean                    | expand_copy                  | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w96:163  | 3   | SIBL    | transpose_copy          | split_with_sizes_copy        | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w96:387  | 0   | SIBL    | mul                     | eq;max                       | 11    | O/O/O | Z/Z/Z | multiOut  |                          |
| w97:413  | 3   | SIBL    | min                     | transpose_copy               | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w98:147  | 4   | SIBL    | prod                    | clone                        | 2     | O/O/O | Z/Z/Z | multiOut  |                          |
| w98:464  | 0   | SIBL    | ceil                    | sqrt                         | 7     | O/O/O | Z/Z/Z | multiOut  |                          |
| w98:507  | 0   | SIBL    | t_copy                  | amax                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w99:145  | 0   | SIBL    | amin                    | var                          | 6     | O/O/O | Z/Z/Z | multiOut  |                          |
| w99:297  | 0   | SIBL    | pow                     | t_copy                       | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w99:613  | 2   | SIBL    | remainder               | unbind_copy                  | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w99:619  | 0   | SIBL    | glu                     | acosh                        | 8     | O/O/O | Z/Z/Z | multiOut  |                          |
| w99:723  | 3   | SIBL    | clamp                   | transpose_copy               | 5     | O/O/O | Z/Z/Z | multiOut  |                          |
| w9:18    | 2   | SIBL    | replication_pad1d       | split_with_sizes_copy        | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w9:251   | 4   | SIBL    | roll                    | amax                         | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
| w9:394   | 0   | SIBL    | fmod                    | transpose_copy               | 3     | O/O/O | Z/Z/Z | multiOut  |                          |
| w9:474   | 1   | SIBL    | prod                    | squeeze_copy                 | 4     | O/O/O | Z/Z/Z | multiOut  |                          |
