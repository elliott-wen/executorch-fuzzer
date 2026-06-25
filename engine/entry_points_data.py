"""entry_points_data.py — embedded symbol→op map for the mobile op set.

Hardcoded subset of entry_points.tsv: only symbols whose op_name is in the mobile
(overload-exact) executorch allowlist. No external data file; self-contained.

Each value: list of (op_name, dispatch_key, demangled) for that mangled C++ symbol.
"""

BY_SYMBOL = {
    '_ZN2at6native10alias_copyERKNS_6TensorE': [
        ('alias_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::alias_copy(at::Tensor const&)'),
    ],
    '_ZN2at6native10arange_outERKN3c106ScalarERNS_6TensorE': [
        ('arange.out', 'CompositeExplicitAutograd', 'at::native::arange_out(c10::Scalar const&, at::Tensor&)'),
        ('arange.start_out', 'CPU', 'at::native::arange_out(c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native10arange_outERKN3c106ScalarES4_S4_RNS_6TensorE': [
        ('arange.out', 'CompositeExplicitAutograd', 'at::native::arange_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)'),
        ('arange.start_out', 'CPU', 'at::native::arange_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native10bitwise_orERKN3c106ScalarERKNS_6TensorE': [
        ('bitwise_or.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_or(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native10bitwise_orERKNS_6TensorERKN3c106ScalarE': [
        ('bitwise_or.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_or(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native10cumsum_outERKNS_6TensorENS_7DimnameESt8optionalIN3c1010ScalarTypeEERS1_': [
        ('cumsum.out', 'CPU', 'at::native::cumsum_out(at::Tensor const&, at::Dimname, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native10gather_outERKNS_6TensorENS_7DimnameES3_bRS1_': [
        ('gather.out', 'CPU', 'at::native::gather_out(at::Tensor const&, at::Dimname, at::Tensor const&, bool, at::Tensor&)'),
    ],
    '_ZN2at6native10linear_outERKNS_6TensorES3_RKSt8optionalIS1_ERS1_': [
        ('linear.out', 'CompositeExplicitAutograd', 'at::native::linear_out(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, at::Tensor&)'),
    ],
    '_ZN2at6native10logical_orERKNS_6TensorES3_': [
        ('logical_or', 'CompositeExplicitAutograd', 'at::native::logical_or(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11bitwise_andERKN3c106ScalarERKNS_6TensorE': [
        ('bitwise_and.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_and(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11bitwise_andERKNS_6TensorERKN3c106ScalarE': [
        ('bitwise_and.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_and(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native11bitwise_xorERKN3c106ScalarERKNS_6TensorE': [
        ('bitwise_xor.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_xor(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11bitwise_xorERKNS_6TensorERKN3c106ScalarE': [
        ('bitwise_xor.Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_xor(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native11convolutionERKNS_6TensorES3_RKSt8optionalIS1_EN3c108ArrayRefIlEESA_SA_bSA_l': [
        ('convolution', 'CompositeExplicitAutograd', 'at::native::convolution(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long)'),
    ],
    '_ZN2at6native11detach_copyERKNS_6TensorE': [
        ('detach_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::detach_copy(at::Tensor const&)'),
    ],
    '_ZN2at6native11logical_andERKNS_6TensorES3_': [
        ('logical_and', 'CompositeExplicitAutograd', 'at::native::logical_and(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11logical_notERKNS_6TensorE': [
        ('logical_not', 'CompositeExplicitAutograd', 'at::native::logical_not(at::Tensor const&)'),
    ],
    '_ZN2at6native11logical_xorERKNS_6TensorES3_': [
        ('logical_xor', 'CompositeExplicitAutograd', 'at::native::logical_xor(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11masked_fillERKNS_6TensorES3_RKN3c106ScalarE': [
        ('masked_fill.Scalar', 'CompositeExplicitAutograd', 'at::native::masked_fill(at::Tensor const&, at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native11masked_fillERKNS_6TensorES3_S3_': [
        ('masked_fill.Scalar', 'CompositeExplicitAutograd', 'at::native::masked_fill(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11nonzero_cpuERKNS_6TensorE': [
        ('nonzero', 'CPU', 'at::native::nonzero_cpu(at::Tensor const&)'),
    ],
    '_ZN2at6native11scatter_addERKNS_6TensorENS_7DimnameES3_S3_': [
        ('scatter_add.out', 'CPU', 'at::native::scatter_add(at::Tensor const&, at::Dimname, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native11unfold_copyERKNS_6TensorElll': [
        ('unfold_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::unfold_copy(at::Tensor const&, long, long, long)'),
    ],
    '_ZN2at6native12_fft_c2r_mklERKNS_6TensorEN3c108ArrayRefIlEEll': [
        ('_fft_c2r', 'CPU', 'at::native::_fft_c2r_mkl(at::Tensor const&, c10::ArrayRef<long>, long, long)'),
    ],
    '_ZN2at6native12_fft_r2c_mklERKNS_6TensorEN3c108ArrayRefIlEElb': [
        ('_fft_r2c', 'CPU', 'at::native::_fft_r2c_mkl(at::Tensor const&, c10::ArrayRef<long>, long, bool)'),
    ],
    '_ZN2at6native12floor_divideERKNS_6TensorERKN3c106ScalarE': [
        ('floor_divide', 'CPU', 'at::native::floor_divide(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native12floor_divideERKNS_6TensorES3_': [
        ('floor_divide', 'CPU', 'at::native::floor_divide(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native12hardtanh_outERKNS_6TensorERKN3c106ScalarES7_RS1_': [
        ('hardtanh.out', 'CPU', 'at::native::hardtanh_out(at::Tensor const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native12permute_copyERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('permute_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::permute_copy(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native13diagonal_copyERKNS_6TensorElll': [
        ('diagonal_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::diagonal_copy(at::Tensor const&, long, long, long)'),
    ],
    '_ZN2at6native13max_unary_outERKNS_6TensorERS1_': [
        ('max.unary_out', 'CPU', 'at::native::max_unary_out(at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native13min_unary_outERKNS_6TensorERS1_': [
        ('min.unary_out', 'CPU', 'at::native::min_unary_out(at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native13remainder_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('remainder.Scalar_out', 'CompositeExplicitAutograd', 'at::native::remainder_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('remainder.Tensor_out', 'CPU', 'at::native::remainder_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native13slice_scatterERKNS_6TensorES3_lSt8optionalIlES5_l': [
        ('slice_scatter', 'CompositeExplicitAutogradNonFunctional', 'at::native::slice_scatter(at::Tensor const&, at::Tensor const&, long, std::optional<long>, std::optional<long>, long)'),
    ],
    '_ZN2at6native14_cdist_forwardERKNS_6TensorES3_dSt8optionalIlE': [
        ('_cdist_forward', 'CPU', 'at::native::_cdist_forward(at::Tensor const&, at::Tensor const&, double, std::optional<long>)'),
    ],
    '_ZN2at6native14_conj_physicalERKNS_6TensorE': [
        ('_conj_physical', 'CompositeExplicitAutograd', 'at::native::_conj_physical(at::Tensor const&)'),
    ],
    '_ZN2at6native14_pdist_forwardERKNS_6TensorEd': [
        ('_pdist_forward', 'CPU', 'at::native::_pdist_forward(at::Tensor const&, double)'),
    ],
    '_ZN2at6native14bitwise_or_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('bitwise_or.Scalar_out', 'CompositeExplicitAutograd', 'at::native::bitwise_or_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('bitwise_or.Tensor_out', 'CPU', 'at::native::bitwise_or_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native14layer_norm_cpuERKNS_6TensorEN3c108ArrayRefIlEERKSt8optionalIS1_ESA_d': [
        ('native_layer_norm', 'CPU', 'at::native::layer_norm_cpu(at::Tensor const&, c10::ArrayRef<long>, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, double)'),
    ],
    '_ZN2at6native14logical_or_outERKNS_6TensorES3_RS1_': [
        ('logical_or.out', 'CPU', 'at::native::logical_or_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native14masked_scatterERKNS_6TensorES3_S3_': [
        ('masked_scatter', 'CompositeExplicitAutograd', 'at::native::masked_scatter(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native14mean_dtype_outERKNS_6TensorESt8optionalIN3c1010ScalarTypeEERS1_': [
        ('mean.dtype_out', 'CompositeExplicitAutograd', 'at::native::mean_dtype_out(at::Tensor const&, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native14unsqueeze_copyERKNS_6TensorEl': [
        ('unsqueeze_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::unsqueeze_copy(at::Tensor const&, long)'),
    ],
    '_ZN2at6native14where_self_outERKNS_6TensorES3_S3_RS1_': [
        ('where.self_out', 'CPU', 'at::native::where_self_out(at::Tensor const&, at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native15bitwise_and_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('bitwise_and.Scalar_out', 'CompositeExplicitAutograd', 'at::native::bitwise_and_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('bitwise_and.Tensor_out', 'CPU', 'at::native::bitwise_and_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native15bitwise_xor_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('bitwise_xor.Scalar_out', 'CompositeExplicitAutograd', 'at::native::bitwise_xor_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('bitwise_xor.Tensor_out', 'CPU', 'at::native::bitwise_xor_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native15constant_pad_ndERKNS_6TensorEN3c108ArrayRefIlEERKNS4_6ScalarE': [
        ('constant_pad_nd', 'CompositeExplicitAutograd', 'at::native::constant_pad_nd(at::Tensor const&, c10::ArrayRef<long>, c10::Scalar const&)'),
    ],
    '_ZN2at6native15lift_fresh_copyERKNS_6TensorE': [
        ('lift_fresh_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::lift_fresh_copy(at::Tensor const&)'),
    ],
    '_ZN2at6native15logical_and_outERKNS_6TensorES3_RS1_': [
        ('logical_and.out', 'CPU', 'at::native::logical_and_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native15logical_not_outERKNS_6TensorERS1_': [
        ('logical_not.out', 'CPU', 'at::native::logical_not_out(at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native15logical_xor_outERKNS_6TensorES3_RS1_': [
        ('logical_xor.out', 'CPU', 'at::native::logical_xor_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native15math_group_normERKNS_6TensorERKSt8optionalIS1_ES7_lllld': [
        ('native_group_norm', 'CompositeExplicitAutograd', 'at::native::math_group_norm(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, long, long, long, long, double)'),
    ],
    '_ZN2at6native15nonzero_out_cpuERKNS_6TensorERS1_': [
        ('nonzero.out', 'CPU', 'at::native::nonzero_out_cpu(at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native15unbind_copy_intERKNS_6TensorEl': [
        ('unbind_copy.int', 'CompositeExplicitAutogradNonFunctional', 'at::native::unbind_copy_int(at::Tensor const&, long)'),
    ],
    '_ZN2at6native16_fft_c2r_mkl_outERKNS_6TensorEN3c108ArrayRefIlEEllRS1_': [
        ('_fft_c2r.out', 'CPU', 'at::native::_fft_c2r_mkl_out(at::Tensor const&, c10::ArrayRef<long>, long, long, at::Tensor&)'),
    ],
    '_ZN2at6native16_fft_r2c_mkl_outERKNS_6TensorEN3c108ArrayRefIlEElbRS1_': [
        ('_fft_r2c.out', 'CPU', 'at::native::_fft_r2c_mkl_out(at::Tensor const&, c10::ArrayRef<long>, long, bool, at::Tensor&)'),
    ],
    '_ZN2at6native16embedding_symintERKNS_6TensorES3_N3c106SymIntEbb': [
        ('embedding', 'CompositeExplicitAutograd', 'at::native::embedding_symint(at::Tensor const&, at::Tensor const&, c10::SymInt, bool, bool)'),
    ],
    '_ZN2at6native16floor_divide_outERKNS_6TensorES3_RS1_': [
        ('floor_divide.out', 'CPU', 'at::native::floor_divide_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native16squeeze_copy_dimERKNS_6TensorEl': [
        ('squeeze_copy.dim', 'CompositeExplicitAutogradNonFunctional', 'at::native::squeeze_copy_dim(at::Tensor const&, long)'),
    ],
    '_ZN2at6native16view_copy_symintERKNS_6TensorEN3c108ArrayRefINS4_6SymIntEEE': [
        ('view_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::view_copy_symint(at::Tensor const&, c10::ArrayRef<c10::SymInt>)'),
    ],
    '_ZN2at6native17index_select_cpu_ERKNS_6TensorElS3_': [
        ('index_select', 'CPU', 'at::native::index_select_cpu_(at::Tensor const&, long, at::Tensor const&)'),
    ],
    '_ZN2at6native17masked_select_cpuERKNS_6TensorES3_': [
        ('masked_select', 'CPU', 'at::native::masked_select_cpu(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native17native_group_normERKNS_6TensorERKSt8optionalIS1_ES7_lllld': [
        ('native_group_norm', 'CPU', 'at::native::native_group_norm(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, long, long, long, long, double)'),
    ],
    '_ZN2at6native17pixel_shuffle_cpuERKNS_6TensorEl': [
        ('pixel_shuffle', 'CPU', 'at::native::pixel_shuffle_cpu(at::Tensor const&, long)'),
    ],
    '_ZN2at6native17squeeze_copy_dimsERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('squeeze_copy.dims', 'CompositeExplicitAutogradNonFunctional', 'at::native::squeeze_copy_dims(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native17view_as_real_copyERKNS_6TensorE': [
        ('view_as_real_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::view_as_real_copy(at::Tensor const&)'),
    ],
    '_ZN2at6native18bitwise_left_shiftERKN3c106ScalarERKNS_6TensorE': [
        ('bitwise_left_shift.Tensor_Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_left_shift(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18bitwise_left_shiftERKNS_6TensorERKN3c106ScalarE': [
        ('bitwise_left_shift.Tensor_Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_left_shift(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native18expand_copy_symintERKNS_6TensorEN3c108ArrayRefINS4_6SymIntEEEb': [
        ('expand_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::expand_copy_symint(at::Tensor const&, c10::ArrayRef<c10::SymInt>, bool)'),
    ],
    '_ZN2at6native18math_pixel_shuffleERKNS_6TensorEl': [
        ('pixel_shuffle', 'CompositeExplicitAutogradNonFunctional', 'at::native::math_pixel_shuffle(at::Tensor const&, long)'),
    ],
    '_ZN2at6native18native_dropout_cpuERKNS_6TensorEdSt8optionalIbE': [
        ('native_dropout', 'CPU', 'at::native::native_dropout_cpu(at::Tensor const&, double, std::optional<bool>)'),
    ],
    '_ZN2at6native18select_copy_symintERKNS_6TensorElN3c106SymIntE': [
        ('select_copy.int', 'CompositeExplicitAutogradNonFunctional', 'at::native::select_copy_symint(at::Tensor const&, long, c10::SymInt)'),
    ],
    '_ZN2at6native18structured_any_out4implERKNS_6TensorElbS4_': [
        ('any.out', 'CPU', 'at::native::structured_any_out::impl(at::Tensor const&, long, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_cos_out4implERKNS_6TensorES4_': [
        ('cos.out', 'CPU', 'at::native::structured_cos_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_div_out4implERKNS_6TensorES4_S4_': [
        ('div.out', 'CPU', 'at::native::structured_div_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_elu_out4implERKNS_6TensorERKN3c106ScalarES8_S8_S4_': [
        ('elu.out', 'CPU', 'at::native::structured_elu_out::impl(at::Tensor const&, c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_erf_out4implERKNS_6TensorES4_': [
        ('erf.out', 'CPU', 'at::native::structured_erf_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_exp_out4implERKNS_6TensorES4_': [
        ('exp.out', 'CPU', 'at::native::structured_exp_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_glu_out4implERKNS_6TensorElS4_': [
        ('glu.out', 'CPU', 'at::native::structured_glu_out::impl(at::Tensor const&, long, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_log_out4implERKNS_6TensorES4_': [
        ('log.out', 'CPU', 'at::native::structured_log_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_max_out4implERKNS_6TensorElbS4_S4_': [
        ('max.dim_max', 'CPU', 'at::native::structured_max_out::impl(at::Tensor const&, long, bool, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_min_out4implERKNS_6TensorElbS4_S4_': [
        ('min.dim_min', 'CPU', 'at::native::structured_min_out::impl(at::Tensor const&, long, bool, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_mul_out4implERKNS_6TensorES4_S4_': [
        ('mul.out', 'CPU', 'at::native::structured_mul_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_neg_out4implERKNS_6TensorES4_': [
        ('neg.out', 'CPU', 'at::native::structured_neg_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_sin_out4implERKNS_6TensorES4_': [
        ('sin.out', 'CPU', 'at::native::structured_sin_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_sub_out4implERKNS_6TensorES4_RKN3c106ScalarES4_': [
        ('sub.out', 'CPU', 'at::native::structured_sub_out::impl(at::Tensor const&, at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_sum_out4implERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS5_10ScalarTypeEES4_': [
        ('sum.IntList_out', 'CPU', 'at::native::structured_sum_out::impl(at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>, at::Tensor const&)'),
    ],
    '_ZN2at6native18structured_tan_out4implERKNS_6TensorES4_': [
        ('tan.out', 'CPU', 'at::native::structured_tan_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native18transpose_copy_intERKNS_6TensorEll': [
        ('transpose_copy.int', 'CompositeExplicitAutogradNonFunctional', 'at::native::transpose_copy_int(at::Tensor const&, long, long)'),
    ],
    '_ZN2at6native19bitwise_right_shiftERKN3c106ScalarERKNS_6TensorE': [
        ('bitwise_right_shift.Tensor_Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_right_shift(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19bitwise_right_shiftERKNS_6TensorERKN3c106ScalarE': [
        ('bitwise_right_shift.Tensor_Scalar', 'CompositeExplicitAutograd', 'at::native::bitwise_right_shift(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native19grid_sampler_2d_cpuERKNS_6TensorES3_llb': [
        ('grid_sampler_2d', 'CPU', 'at::native::grid_sampler_2d_cpu(at::Tensor const&, at::Tensor const&, long, long, bool)'),
    ],
    '_ZN2at6native19pixel_unshuffle_cpuERKNS_6TensorEl': [
        ('pixel_unshuffle', 'CPU', 'at::native::pixel_unshuffle_cpu(at::Tensor const&, long)'),
    ],
    '_ZN2at6native19structured_acos_out4implERKNS_6TensorES4_': [
        ('acos.out', 'CPU', 'at::native::structured_acos_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_amax_out4implERKNS_6TensorEN3c108ArrayRefIlEEbS4_': [
        ('amax.out', 'CPU', 'at::native::structured_amax_out::impl(at::Tensor const&, c10::ArrayRef<long>, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_amin_out4implERKNS_6TensorEN3c108ArrayRefIlEEbS4_': [
        ('amin.out', 'CPU', 'at::native::structured_amin_out::impl(at::Tensor const&, c10::ArrayRef<long>, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_asin_out4implERKNS_6TensorES4_': [
        ('asin.out', 'CPU', 'at::native::structured_asin_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_atan_out4implERKNS_6TensorES4_': [
        ('atan.out', 'CPU', 'at::native::structured_atan_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_ceil_out4implERKNS_6TensorES4_': [
        ('ceil.out', 'CPU', 'at::native::structured_ceil_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_cosh_out4implERKNS_6TensorES4_': [
        ('cosh.out', 'CPU', 'at::native::structured_cosh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_fmod_out4implERKNS_6TensorES4_S4_': [
        ('fmod.Tensor_out', 'CPU', 'at::native::structured_fmod_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_log2_out4implERKNS_6TensorES4_': [
        ('log2.out', 'CPU', 'at::native::structured_log2_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_mean_out4implERKNS_6TensorEN3c1016OptionalArrayRefIlEEbSt8optionalINS5_10ScalarTypeEES4_': [
        ('mean.out', 'CPU', 'at::native::structured_mean_out::impl(at::Tensor const&, c10::OptionalArrayRef<long>, bool, std::optional<c10::ScalarType>, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_prod_out4implERKNS_6TensorElbSt8optionalIN3c1010ScalarTypeEES4_': [
        ('prod.int_out', 'CPU', 'at::native::structured_prod_out::impl(at::Tensor const&, long, bool, std::optional<c10::ScalarType>, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_sign_out4implERKNS_6TensorES4_': [
        ('sign.out', 'CPU', 'at::native::structured_sign_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_sinh_out4implERKNS_6TensorES4_': [
        ('sinh.out', 'CPU', 'at::native::structured_sinh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_sqrt_out4implERKNS_6TensorES4_': [
        ('sqrt.out', 'CPU', 'at::native::structured_sqrt_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_tanh_out4implERKNS_6TensorES4_': [
        ('tanh.out', 'CPU', 'at::native::structured_tanh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native19structured_tril_cpu4implERKNS_6TensorElS4_': [
        ('tril.out', 'CPU', 'at::native::structured_tril_cpu::impl(at::Tensor const&, long, at::Tensor const&)'),
    ],
    '_ZN2at6native19unbind_copy_int_outERKNS_6TensorElN3c108ArrayRefIS1_EE': [
        ('unbind_copy.int_out', 'CompositeExplicitAutograd', 'at::native::unbind_copy_int_out(at::Tensor const&, long, c10::ArrayRef<at::Tensor>)'),
    ],
    '_ZN2at6native20any_dims_out_defaultERKNS_6TensorEN3c1016OptionalArrayRefIlEEbRS1_': [
        ('any.dims_out', 'CompositeExplicitAutograd', 'at::native::any_dims_out_default(at::Tensor const&, c10::OptionalArrayRef<long>, bool, at::Tensor&)'),
    ],
    '_ZN2at6native20convolution_backwardERKNS_6TensorES3_S3_N3c1016OptionalArrayRefIlEENS4_8ArrayRefIlEES8_S8_bS8_lSt5arrayIbLm3EE': [
        ('convolution_backward', 'CompositeExplicitAutograd', 'at::native::convolution_backward(at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::OptionalArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long, std::array<bool, 3ul>)'),
    ],
    '_ZN2at6native20math_pixel_unshuffleERKNS_6TensorEl': [
        ('pixel_unshuffle', 'CompositeExplicitAutogradNonFunctional', 'at::native::math_pixel_unshuffle(at::Tensor const&, long)'),
    ],
    '_ZN2at6native20reflection_pad2d_cpuERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('reflection_pad2d', 'CPU', 'at::native::reflection_pad2d_cpu(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native20structured_acosh_out4implERKNS_6TensorES4_': [
        ('acosh.out', 'CPU', 'at::native::structured_acosh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_asinh_out4implERKNS_6TensorES4_': [
        ('asinh.out', 'CPU', 'at::native::structured_asinh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_atan2_out4implERKNS_6TensorES4_S4_': [
        ('atan2.out', 'CPU', 'at::native::structured_atan2_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_atanh_out4implERKNS_6TensorES4_': [
        ('atanh.out', 'CPU', 'at::native::structured_atanh_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_clamp_out4implERKNS_6TensorEN3c1011OptionalRefINS5_6ScalarEEES8_S4_': [
        ('clamp.out', 'CPU', 'at::native::structured_clamp_out::impl(at::Tensor const&, c10::OptionalRef<c10::Scalar>, c10::OptionalRef<c10::Scalar>, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_expm1_out4implERKNS_6TensorES4_': [
        ('expm1.out', 'CPU', 'at::native::structured_expm1_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_floor_out4implERKNS_6TensorES4_': [
        ('floor.out', 'CPU', 'at::native::structured_floor_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_index_out4implERKNS_6TensorEN3c1011SmallVectorIlLj5EEES7_S4_': [
        ('index.Tensor_out', 'CPU', 'at::native::structured_index_out::impl(at::Tensor const&, c10::SmallVector<long, 5u>, c10::SmallVector<long, 5u>, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_log10_out4implERKNS_6TensorES4_': [
        ('log10.out', 'CPU', 'at::native::structured_log10_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_log1p_out4implERKNS_6TensorES4_': [
        ('log1p.out', 'CPU', 'at::native::structured_log1p_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_round_out4implERKNS_6TensorES4_': [
        ('round.out', 'CPU', 'at::native::structured_round_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_rsqrt_out4implERKNS_6TensorES4_': [
        ('rsqrt.out', 'CPU', 'at::native::structured_rsqrt_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native20structured_trunc_out4implERKNS_6TensorES4_': [
        ('trunc.out', 'CPU', 'at::native::structured_trunc_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native21_batch_norm_legit_cpuERKNS_6TensorERKSt8optionalIS1_ES7_RS1_S8_bdd': [
        ('_native_batch_norm_legit', 'CPU', 'at::native::_batch_norm_legit_cpu(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, at::Tensor&, at::Tensor&, bool, double, double)'),
    ],
    '_ZN2at6native21index_select_out_cpu_ERKNS_6TensorElS3_RS1_': [
        ('index_select.out', 'CPU', 'at::native::index_select_out_cpu_(at::Tensor const&, long, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native21masked_select_out_cpuERKNS_6TensorES3_RS1_': [
        ('masked_select.out', 'CPU', 'at::native::masked_select_out_cpu(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native21narrow_copy_dense_cpuERKNS_6TensorElll': [
        ('narrow_copy', 'CPU', 'at::native::narrow_copy_dense_cpu(at::Tensor const&, long, long, long)'),
    ],
    '_ZN2at6native21repeat_interleave_cpuERKNS_6TensorESt8optionalIlE': [
        ('repeat_interleave.Tensor', 'CPU', 'at::native::repeat_interleave_cpu(at::Tensor const&, std::optional<long>)'),
    ],
    '_ZN2at6native21select_scatter_symintERKNS_6TensorES3_lN3c106SymIntE': [
        ('select_scatter', 'CompositeExplicitAutogradNonFunctional', 'at::native::select_scatter_symint(at::Tensor const&, at::Tensor const&, long, c10::SymInt)'),
    ],
    '_ZN2at6native21split_copy_Tensor_outERKNS_6TensorEllN3c108ArrayRefIS1_EE': [
        ('split_copy.Tensor_out', 'CompositeExplicitAutograd', 'at::native::split_copy_Tensor_out(at::Tensor const&, long, long, c10::ArrayRef<at::Tensor>)'),
    ],
    '_ZN2at6native21structured_argmax_out4implERKNS_6TensorESt8optionalIlEbS4_': [
        ('argmax.out', 'CPU', 'at::native::structured_argmax_out::impl(at::Tensor const&, std::optional<long>, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native21structured_argmin_out4implERKNS_6TensorESt8optionalIlEbS4_': [
        ('argmin.out', 'CPU', 'at::native::structured_argmin_out::impl(at::Tensor const&, std::optional<long>, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native21structured_cumsum_out4implERKNS_6TensorElSt8optionalIN3c1010ScalarTypeEES4_': [
        ('cumsum.out', 'CPU', 'at::native::structured_cumsum_out::impl(at::Tensor const&, long, std::optional<c10::ScalarType>, at::Tensor const&)'),
    ],
    '_ZN2at6native21structured_gather_out4implERKNS_6TensorElS4_bS4_': [
        ('gather.out', 'CPU', 'at::native::structured_gather_out::impl(at::Tensor const&, long, at::Tensor const&, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native21structured_mm_out_cpu4implERKNS_6TensorES4_S4_': [
        ('mm.out', 'CPU', 'at::native::structured_mm_out_cpu::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22as_strided_copy_symintERKNS_6TensorEN3c108ArrayRefINS4_6SymIntEEES7_St8optionalIS6_E': [
        ('as_strided_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::as_strided_copy_symint(at::Tensor const&, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::SymInt>)'),
    ],
    '_ZN2at6native22bitwise_left_shift_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('bitwise_left_shift.Tensor_Scalar_out', 'CompositeExplicitAutograd', 'at::native::bitwise_left_shift_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('bitwise_left_shift.Tensor_out', 'CPU', 'at::native::bitwise_left_shift_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native22math_native_layer_normERKNS_6TensorEN3c108ArrayRefIlEERKSt8optionalIS1_ESA_d': [
        ('native_layer_norm', 'CompositeExplicitAutograd', 'at::native::math_native_layer_norm(at::Tensor const&, c10::ArrayRef<long>, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, double)'),
    ],
    '_ZN2at6native22structured_any_all_out4implERKNS_6TensorES4_': [
        ('any.all_out', 'CPU', 'at::native::structured_any_all_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_bmm_out_cpu4implERKNS_6TensorES4_S4_': [
        ('bmm.out', 'CPU', 'at::native::structured_bmm_out_cpu::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_cat_out_cpu4implERKN3c108IListRefINS_6TensorEEEllbbbNS2_12MemoryFormatERKS4_': [
        ('cat.out', 'CPU', 'at::native::structured_cat_out_cpu::impl(c10::IListRef<at::Tensor> const&, long, long, bool, bool, bool, c10::MemoryFormat, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_maximum_out4implERKNS_6TensorES4_S4_': [
        ('maximum.out', 'CPU', 'at::native::structured_maximum_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_minimum_out4implERKNS_6TensorES4_S4_': [
        ('minimum.out', 'CPU', 'at::native::structured_minimum_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_scatter_add4implERKNS_6TensorElS4_S4_S4_': [
        ('scatter_add.out', 'CPU', 'at::native::structured_scatter_add::impl(at::Tensor const&, long, at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native22structured_sigmoid_out4implERKNS_6TensorES4_': [
        ('sigmoid.out', 'CPU', 'at::native::structured_sigmoid_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native23adaptive_avg_pool2d_cpuERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('_adaptive_avg_pool2d', 'CPU', 'at::native::adaptive_avg_pool2d_cpu(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native23bitwise_right_shift_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('bitwise_right_shift.Tensor_Scalar_out', 'CompositeExplicitAutograd', 'at::native::bitwise_right_shift_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('bitwise_right_shift.Tensor_out', 'CPU', 'at::native::bitwise_right_shift_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native23structured_any_dims_out4implERKNS_6TensorEN3c1016OptionalArrayRefIlEEbS4_': [
        ('any.dims_out', 'CPU', 'at::native::structured_any_dims_out::impl(at::Tensor const&, c10::OptionalArrayRef<long>, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native23structured_div_out_mode4implERKNS_6TensorES4_St8optionalISt17basic_string_viewIcSt11char_traitsIcEEES4_': [
        ('div.out_mode', 'CPU', 'at::native::structured_div_out_mode::impl(at::Tensor const&, at::Tensor const&, std::optional<std::basic_string_view<char, std::char_traits<char> > >, at::Tensor const&)'),
    ],
    '_ZN2at6native23structured_gelu_out_cpu4implERKNS_6TensorESt17basic_string_viewIcSt11char_traitsIcEES4_': [
        ('gelu.out', 'CPU', 'at::native::structured_gelu_out_cpu::impl(at::Tensor const&, std::basic_string_view<char, std::char_traits<char> >, at::Tensor const&)'),
    ],
    '_ZN2at6native23structured_topk_out_cpu4implERKNS_6TensorEllbbS4_S4_': [
        ('topk.values', 'CPU', 'at::native::structured_topk_out_cpu::impl(at::Tensor const&, long, long, bool, bool, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24narrow_copy_dense_symintERKNS_6TensorElN3c106SymIntES5_': [
        ('narrow_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::narrow_copy_dense_symint(at::Tensor const&, long, c10::SymInt, c10::SymInt)'),
    ],
    '_ZN2at6native24reflection_pad2d_out_cpuERKNS_6TensorEN3c108ArrayRefIlEERS1_': [
        ('reflection_pad2d.out', 'CPU', 'at::native::reflection_pad2d_out_cpu(at::Tensor const&, c10::ArrayRef<long>, at::Tensor&)'),
    ],
    '_ZN2at6native24slice_copy_Tensor_symintERKNS_6TensorElSt8optionalIN3c106SymIntEES7_S6_': [
        ('slice_copy.Tensor', 'CompositeExplicitAutogradNonFunctional', 'at::native::slice_copy_Tensor_symint(at::Tensor const&, long, std::optional<c10::SymInt>, std::optional<c10::SymInt>, c10::SymInt)'),
    ],
    '_ZN2at6native24split_copy_Tensor_symintERKNS_6TensorEN3c106SymIntEl': [
        ('split_copy.Tensor', 'CompositeExplicitAutogradNonFunctional', 'at::native::split_copy_Tensor_symint(at::Tensor const&, c10::SymInt, long)'),
    ],
    '_ZN2at6native24structured_addmm_out_cpu4implERKNS_6TensorES4_S4_RKN3c106ScalarES8_S4_': [
        ('addmm.out', 'CPU', 'at::native::structured_addmm_out_cpu::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_eq_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('eq.Scalar_out', 'CPU', 'at::native::structured_eq_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_eq_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('eq.Tensor_out', 'CPU', 'at::native::structured_eq_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_ge_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('ge.Scalar_out', 'CPU', 'at::native::structured_ge_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_ge_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('ge.Tensor_out', 'CPU', 'at::native::structured_ge_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_gt_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('gt.Scalar_out', 'CPU', 'at::native::structured_gt_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_gt_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('gt.Tensor_out', 'CPU', 'at::native::structured_gt_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_le_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('le.Scalar_out', 'CPU', 'at::native::structured_le_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_le_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('le.Tensor_out', 'CPU', 'at::native::structured_le_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_lt_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('lt.Scalar_out', 'CPU', 'at::native::structured_lt_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_lt_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('lt.Tensor_out', 'CPU', 'at::native::structured_lt_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_ne_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('ne.Scalar_out', 'CPU', 'at::native::structured_ne_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_ne_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('ne.Tensor_out', 'CPU', 'at::native::structured_ne_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native24structured_remainder_out4implERKNS_6TensorES4_S4_': [
        ('remainder.Tensor_out', 'CPU', 'at::native::structured_remainder_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native25_batch_norm_legit_cpu_outERKNS_6TensorERKSt8optionalIS1_ES7_RS1_S8_bddS8_S8_S8_': [
        ('_native_batch_norm_legit.out', 'CPU', 'at::native::_batch_norm_legit_cpu_out(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, at::Tensor&, at::Tensor&, bool, double, double, at::Tensor&, at::Tensor&, at::Tensor&)'),
    ],
    '_ZN2at6native25narrow_copy_dense_cpu_outERKNS_6TensorElllRS1_': [
        ('narrow_copy.out', 'CPU', 'at::native::narrow_copy_dense_cpu_out(at::Tensor const&, long, long, long, at::Tensor&)'),
    ],
    '_ZN2at6native25split_with_sizes_copy_outERKNS_6TensorEN3c108ArrayRefIlEElNS5_IS1_EE': [
        ('split_with_sizes_copy.out', 'CompositeExplicitAutograd', 'at::native::split_with_sizes_copy_out(at::Tensor const&, c10::ArrayRef<long>, long, c10::ArrayRef<at::Tensor>)'),
    ],
    '_ZN2at6native25structured_bitwise_or_out4implERKNS_6TensorES4_S4_': [
        ('bitwise_or.Tensor_out', 'CPU', 'at::native::structured_bitwise_or_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native25structured_leaky_relu_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('leaky_relu.out', 'CPU', 'at::native::structured_leaky_relu_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native25structured_pow_Scalar_out4implERKN3c106ScalarERKNS_6TensorES8_': [
        ('pow.Scalar_out', 'CPU', 'at::native::structured_pow_Scalar_out::impl(c10::Scalar const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native25structured_reciprocal_out4implERKNS_6TensorES4_': [
        ('reciprocal.out', 'CPU', 'at::native::structured_reciprocal_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native26structured_bitwise_and_out4implERKNS_6TensorES4_S4_': [
        ('bitwise_and.Tensor_out', 'CPU', 'at::native::structured_bitwise_and_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native26structured_bitwise_not_out4implERKNS_6TensorES4_': [
        ('bitwise_not.out', 'CPU', 'at::native::structured_bitwise_not_out::impl(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native26structured_bitwise_xor_out4implERKNS_6TensorES4_S4_': [
        ('bitwise_xor.Tensor_out', 'CPU', 'at::native::structured_bitwise_xor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native26structured_scatter_src_out4implERKNS_6TensorElS4_S4_S4_': [
        ('scatter.src_out', 'CPU', 'at::native::structured_scatter_src_out::impl(at::Tensor const&, long, at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native26structured_softmax_cpu_out4implERKNS_6TensorElbS4_': [
        ('_softmax.out', 'CPU', 'at::native::structured_softmax_cpu_out::impl(at::Tensor const&, long, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native27structured_clamp_Tensor_out4implERKNS_6TensorENS_17OptionalTensorRefES5_S4_': [
        ('clamp.Tensor_out', 'CPU', 'at::native::structured_clamp_Tensor_out::impl(at::Tensor const&, at::OptionalTensorRef, at::OptionalTensorRef, at::Tensor const&)'),
    ],
    '_ZN2at6native28split_with_sizes_copy_symintERKNS_6TensorEN3c108ArrayRefINS4_6SymIntEEEl': [
        ('split_with_sizes_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::split_with_sizes_copy_symint(at::Tensor const&, c10::ArrayRef<c10::SymInt>, long)'),
    ],
    '_ZN2at6native28structured_scatter_value_out4implERKNS_6TensorElS4_RKN3c106ScalarES4_': [
        ('scatter.value_out', 'CPU', 'at::native::structured_scatter_value_out::impl(at::Tensor const&, long, at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native29_batch_norm_legit_no_trainingERKNS_6TensorERKSt8optionalIS1_ES7_S3_S3_dd': [
        ('_native_batch_norm_legit_no_training', 'CompositeExplicitAutograd', 'at::native::_batch_norm_legit_no_training(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, at::Tensor const&, at::Tensor const&, double, double)'),
    ],
    '_ZN2at6native29structured_avg_pool2d_out_cpu4implERKNS_6TensorEllllllbbSt8optionalIlES4_': [
        ('avg_pool2d.out', 'CPU', 'at::native::structured_avg_pool2d_out_cpu::impl(at::Tensor const&, long, long, long, long, long, long, bool, bool, std::optional<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native30_batch_norm_legit_no_stats_cpuERKNS_6TensorERKSt8optionalIS1_ES7_bdd': [
        ('_native_batch_norm_legit.no_stats', 'CPU', 'at::native::_batch_norm_legit_no_stats_cpu(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, bool, double, double)'),
    ],
    '_ZN2at6native30structured_log_softmax_cpu_out4implERKNS_6TensorElbS4_': [
        ('_log_softmax.out', 'CPU', 'at::native::structured_log_softmax_cpu_out::impl(at::Tensor const&, long, bool, at::Tensor const&)'),
    ],
    '_ZN2at6native32structured_pow_Tensor_Scalar_out4implERKNS_6TensorERKN3c106ScalarES4_': [
        ('pow.Tensor_Scalar_out', 'CPU', 'at::native::structured_pow_Tensor_Scalar_out::impl(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native32structured_pow_Tensor_Tensor_out4implERKNS_6TensorES4_S4_': [
        ('pow.Tensor_Tensor_out', 'CPU', 'at::native::structured_pow_Tensor_Tensor_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native33structured_bitwise_left_shift_out4implERKNS_6TensorES4_S4_': [
        ('bitwise_left_shift.Tensor_out', 'CPU', 'at::native::structured_bitwise_left_shift_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native34_batch_norm_legit_no_stats_cpu_outERKNS_6TensorERKSt8optionalIS1_ES7_bddRS1_S8_S8_': [
        ('_native_batch_norm_legit.no_stats_out', 'CPU', 'at::native::_batch_norm_legit_no_stats_cpu_out(at::Tensor const&, std::optional<at::Tensor> const&, std::optional<at::Tensor> const&, bool, double, double, at::Tensor&, at::Tensor&, at::Tensor&)'),
    ],
    '_ZN2at6native34structured_bitwise_right_shift_out4implERKNS_6TensorES4_S4_': [
        ('bitwise_right_shift.Tensor_out', 'CPU', 'at::native::structured_bitwise_right_shift_out::impl(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native35structured_reflection_pad1d_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES4_': [
        ('reflection_pad1d.out', 'CPU', 'at::native::structured_reflection_pad1d_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native35structured_reflection_pad3d_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES4_': [
        ('reflection_pad3d.out', 'CPU', 'at::native::structured_reflection_pad3d_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native36structured_replication_pad1d_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES4_': [
        ('replication_pad1d.out', 'CPU', 'at::native::structured_replication_pad1d_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native36structured_replication_pad2d_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES4_': [
        ('replication_pad2d.out', 'CPU', 'at::native::structured_replication_pad2d_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native36structured_replication_pad3d_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES4_': [
        ('replication_pad3d.out', 'CPU', 'at::native::structured_replication_pad3d_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, at::Tensor const&)'),
    ],
    '_ZN2at6native3absERKNS_6TensorE': [
        ('abs', 'CompositeExplicitAutograd', 'at::native::abs(at::Tensor const&)'),
    ],
    '_ZN2at6native3addERKNS_6TensorERKN3c106ScalarES7_': [
        ('add.Scalar', 'CompositeExplicitAutograd', 'at::native::add(at::Tensor const&, c10::Scalar const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native3divERKNS_6TensorERKN3c106ScalarE': [
        ('div.Scalar', 'CompositeExplicitAutograd', 'at::native::div(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native3divERKNS_6TensorERKN3c106ScalarESt8optionalISt17basic_string_viewIcSt11char_traitsIcEEE': [
        ('div.Scalar', 'CompositeExplicitAutograd', 'at::native::div(at::Tensor const&, c10::Scalar const&, std::optional<std::basic_string_view<char, std::char_traits<char> > >)'),
    ],
    '_ZN2at6native3maxERKNS_6TensorE': [
        ('max', 'CPU', 'at::native::max(at::Tensor const&)'),
    ],
    '_ZN2at6native3maxERKNS_6TensorENS_7DimnameEb': [
        ('max', 'CPU', 'at::native::max(at::Tensor const&, at::Dimname, bool)'),
    ],
    '_ZN2at6native3maxERKNS_6TensorES3_': [
        ('max', 'CPU', 'at::native::max(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native3minERKNS_6TensorE': [
        ('min', 'CPU', 'at::native::min(at::Tensor const&)'),
    ],
    '_ZN2at6native3minERKNS_6TensorENS_7DimnameEb': [
        ('min', 'CPU', 'at::native::min(at::Tensor const&, at::Dimname, bool)'),
    ],
    '_ZN2at6native3minERKNS_6TensorES3_': [
        ('min', 'CPU', 'at::native::min(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native3mulERKNS_6TensorERKN3c106ScalarE': [
        ('mul.Scalar', 'CompositeExplicitAutograd', 'at::native::mul(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native3subERKNS_6TensorERKN3c106ScalarES7_': [
        ('sub.Scalar', 'CompositeExplicitAutograd', 'at::native::sub(at::Tensor const&, c10::Scalar const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native3varERKNS_6TensorEN3c1016OptionalArrayRefIlEERKSt8optionalINS4_6ScalarEEb': [
        ('var.correction', 'CPU', 'at::native::var(at::Tensor const&, c10::OptionalArrayRef<long>, std::optional<c10::Scalar> const&, bool)'),
    ],
    '_ZN2at6native3varERKNS_6TensorEN3c1016OptionalArrayRefIlEEbb': [
        ('var.correction', 'CPU', 'at::native::var(at::Tensor const&, c10::OptionalArrayRef<long>, bool, bool)'),
    ],
    '_ZN2at6native3varERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEERKSt8optionalINS4_6ScalarEEb': [
        ('var.correction', 'CPU', 'at::native::var(at::Tensor const&, c10::ArrayRef<at::Dimname>, std::optional<c10::Scalar> const&, bool)'),
    ],
    '_ZN2at6native3varERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbb': [
        ('var.correction', 'CPU', 'at::native::var(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, bool)'),
    ],
    '_ZN2at6native3varERKNS_6TensorEb': [
        ('var.correction', 'CPU', 'at::native::var(at::Tensor const&, bool)'),
    ],
    '_ZN2at6native42structured__upsample_bilinear2d_aa_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEEbSt8optionalIdES9_S4_': [
        ('_upsample_bilinear2d_aa.out', 'CPU', 'at::native::structured__upsample_bilinear2d_aa_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, bool, std::optional<double>, std::optional<double>, at::Tensor const&)'),
    ],
    '_ZN2at6native42structured_max_pool2d_with_indices_out_cpu4implERKNS_6TensorEN3c108ArrayRefIlEES7_S7_S7_bS4_S4_': [
        ('max_pool2d_with_indices.out', 'CPU', 'at::native::structured_max_pool2d_with_indices_out_cpu::impl(at::Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native4copyERKNS_6TensorES3_b': [
        ('copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::copy(at::Tensor const&, at::Tensor const&, bool)'),
    ],
    '_ZN2at6native4fillERKNS_6TensorERKN3c106ScalarE': [
        ('fill.Scalar', 'CompositeExplicitAutograd', 'at::native::fill(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native4fillERKNS_6TensorES3_': [
        ('fill.Scalar', 'CompositeExplicitAutograd', 'at::native::fill(at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native4flipERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('flip', 'CPU', 'at::native::flip(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native4fmodERKNS_6TensorERKN3c106ScalarE': [
        ('fmod.Scalar', 'CompositeExplicitAutograd', 'at::native::fmod(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native4meanERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbSt8optionalINS4_10ScalarTypeEE': [
        ('mean', 'CompositeExplicitAutograd', 'at::native::mean(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, std::optional<c10::ScalarType>)'),
    ],
    '_ZN2at6native4meanERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE': [
        ('mean', 'CompositeExplicitAutograd', 'at::native::mean(at::Tensor const&, std::optional<c10::ScalarType>)'),
    ],
    '_ZN2at6native4prodERKNS_6TensorENS_7DimnameEbSt8optionalIN3c1010ScalarTypeEE': [
        ('prod', 'CPU', 'at::native::prod(at::Tensor const&, at::Dimname, bool, std::optional<c10::ScalarType>)'),
    ],
    '_ZN2at6native4prodERKNS_6TensorESt8optionalIN3c1010ScalarTypeEE': [
        ('prod', 'CPU', 'at::native::prod(at::Tensor const&, std::optional<c10::ScalarType>)'),
    ],
    '_ZN2at6native4reluERKNS_6TensorE': [
        ('relu', 'CPU', 'at::native::relu(at::Tensor const&)'),
    ],
    '_ZN2at6native4rollERKNS_6TensorEN3c108ArrayRefIlEES6_': [
        ('roll', 'CPU', 'at::native::roll(at::Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native4rsubERKNS_6TensorERKN3c106ScalarES7_': [
        ('rsub.Scalar', 'CompositeExplicitAutograd', 'at::native::rsub(at::Tensor const&, c10::Scalar const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native4rsubERKNS_6TensorES3_RKN3c106ScalarE': [
        ('rsub.Scalar', 'CompositeExplicitAutograd', 'at::native::rsub(at::Tensor const&, at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native51structured_max_pool2d_with_indices_backward_out_cpu4implERKNS_6TensorES4_N3c108ArrayRefIlEES7_S7_S7_bS4_S4_': [
        ('max_pool2d_with_indices_backward.grad_input', 'CPU', 'at::native::structured_max_pool2d_with_indices_backward_out_cpu::impl(at::Tensor const&, at::Tensor const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native5cloneERKNS_6TensorESt8optionalIN3c1012MemoryFormatEE': [
        ('clone', 'CompositeExplicitAutograd', 'at::native::clone(at::Tensor const&, std::optional<c10::MemoryFormat>)'),
    ],
    '_ZN2at6native5isinfERKNS_6TensorE': [
        ('isinf', 'CompositeExplicitAutograd', 'at::native::isinf(at::Tensor const&)'),
    ],
    '_ZN2at6native5isnanERKNS_6TensorE': [
        ('isnan', 'CPU', 'at::native::isnan(at::Tensor const&)'),
    ],
    '_ZN2at6native5logitERKNS_6TensorESt8optionalIdE': [
        ('logit', 'CPU', 'at::native::logit(at::Tensor const&, std::optional<double>)'),
    ],
    '_ZN2at6native5stackEN3c108ArrayRefINS_6TensorEEEl': [
        ('stack', 'CompositeExplicitAutograd', 'at::native::stack(c10::ArrayRef<at::Tensor>, long)'),
    ],
    '_ZN2at6native5whereERKNS_6TensorE': [
        ('where.self', 'CPU', 'at::native::where(at::Tensor const&)'),
    ],
    '_ZN2at6native5whereERKNS_6TensorERKN3c106ScalarES3_': [
        ('where.self', 'CPU', 'at::native::where(at::Tensor const&, c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native5whereERKNS_6TensorERKN3c106ScalarES7_': [
        ('where.self', 'CPU', 'at::native::where(at::Tensor const&, c10::Scalar const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native5whereERKNS_6TensorES3_RKN3c106ScalarE': [
        ('where.self', 'CPU', 'at::native::where(at::Tensor const&, at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native5whereERKNS_6TensorES3_S3_': [
        ('where.self', 'CPU', 'at::native::where(at::Tensor const&, at::Tensor const&, at::Tensor const&)'),
    ],
    '_ZN2at6native6linearERKNS_6TensorES3_RKSt8optionalIS1_E': [
        ('linear', 'CompositeImplicitAutograd', 'at::native::linear(at::Tensor const&, at::Tensor const&, std::optional<at::Tensor> const&)'),
    ],
    '_ZN2at6native6repeatERKNS_6TensorEN3c108ArrayRefIlEE': [
        ('repeat', 'CompositeExplicitAutograd', 'at::native::repeat(at::Tensor const&, c10::ArrayRef<long>)'),
    ],
    '_ZN2at6native6t_copyERKNS_6TensorE': [
        ('t_copy', 'CompositeExplicitAutogradNonFunctional', 'at::native::t_copy(at::Tensor const&)'),
    ],
    '_ZN2at6native7abs_outERKNS_6TensorERS1_': [
        ('abs.out', 'CPU', 'at::native::abs_out(at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native7any_outERKNS_6TensorENS_7DimnameEbRS1_': [
        ('any.out', 'CPU', 'at::native::any_out(at::Tensor const&, at::Dimname, bool, at::Tensor&)'),
    ],
    '_ZN2at6native7max_outERKNS_6TensorENS_7DimnameEbRS1_S5_': [
        ('max.dim_max', 'CPU', 'at::native::max_out(at::Tensor const&, at::Dimname, bool, at::Tensor&, at::Tensor&)'),
    ],
    '_ZN2at6native7max_outERKNS_6TensorES3_RS1_': [
        ('max.dim_max', 'CPU', 'at::native::max_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native7min_outERKNS_6TensorENS_7DimnameEbRS1_S5_': [
        ('min.dim_min', 'CPU', 'at::native::min_out(at::Tensor const&, at::Dimname, bool, at::Tensor&, at::Tensor&)'),
    ],
    '_ZN2at6native7min_outERKNS_6TensorES3_RS1_': [
        ('min.dim_min', 'CPU', 'at::native::min_out(at::Tensor const&, at::Tensor const&, at::Tensor&)'),
    ],
    '_ZN2at6native7sum_outERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbSt8optionalINS4_10ScalarTypeEERS1_': [
        ('sum.IntList_out', 'CPU', 'at::native::sum_out(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native7sum_outERKNS_6TensorESt8optionalIN3c1010ScalarTypeEERS1_': [
        ('sum.IntList_out', 'CPU', 'at::native::sum_out(at::Tensor const&, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native7var_outERKNS_6TensorEN3c1016OptionalArrayRefIlEERKSt8optionalINS4_6ScalarEEbRS1_': [
        ('var.correction_out', 'CPU', 'at::native::var_out(at::Tensor const&, c10::OptionalArrayRef<long>, std::optional<c10::Scalar> const&, bool, at::Tensor&)'),
    ],
    '_ZN2at6native7var_outERKNS_6TensorEN3c1016OptionalArrayRefIlEEbbRS1_': [
        ('var.correction_out', 'CPU', 'at::native::var_out(at::Tensor const&, c10::OptionalArrayRef<long>, bool, bool, at::Tensor&)'),
    ],
    '_ZN2at6native7var_outERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEERKSt8optionalINS4_6ScalarEEbRS1_': [
        ('var.correction_out', 'CPU', 'at::native::var_out(at::Tensor const&, c10::ArrayRef<at::Dimname>, std::optional<c10::Scalar> const&, bool, at::Tensor&)'),
    ],
    '_ZN2at6native7var_outERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbbRS1_': [
        ('var.correction_out', 'CPU', 'at::native::var_out(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, bool, at::Tensor&)'),
    ],
    '_ZN2at6native8fmod_outERKNS_6TensorERKN3c106ScalarERS1_': [
        ('fmod.Scalar_out', 'CompositeExplicitAutograd', 'at::native::fmod_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
        ('fmod.Tensor_out', 'CPU', 'at::native::fmod_out(at::Tensor const&, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native8full_outEN3c108ArrayRefIlEERKNS1_6ScalarERNS_6TensorE': [
        ('full.out', 'CompositeExplicitAutograd', 'at::native::full_out(c10::ArrayRef<long>, c10::Scalar const&, at::Tensor&)'),
    ],
    '_ZN2at6native8hardtanhERKNS_6TensorERKN3c106ScalarES7_': [
        ('hardtanh', 'CPU', 'at::native::hardtanh(at::Tensor const&, c10::Scalar const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native8mean_outERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbSt8optionalINS4_10ScalarTypeEERS1_': [
        ('mean.out', 'CPU', 'at::native::mean_out(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native8ones_outEN3c108ArrayRefIlEERNS_6TensorE': [
        ('ones.out', 'CompositeExplicitAutograd', 'at::native::ones_out(c10::ArrayRef<long>, at::Tensor&)'),
    ],
    '_ZN2at6native8prod_outERKNS_6TensorENS_7DimnameEbSt8optionalIN3c1010ScalarTypeEERS1_': [
        ('prod.int_out', 'CPU', 'at::native::prod_out(at::Tensor const&, at::Dimname, bool, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native8prod_outERKNS_6TensorESt8optionalIN3c1010ScalarTypeEERS1_': [
        ('prod.int_out', 'CPU', 'at::native::prod_out(at::Tensor const&, std::optional<c10::ScalarType>, at::Tensor&)'),
    ],
    '_ZN2at6native8rand_outEN3c108ArrayRefIlEERNS_6TensorE': [
        ('rand.out', 'CompositeExplicitAutograd', 'at::native::rand_out(c10::ArrayRef<long>, at::Tensor&)'),
    ],
    '_ZN2at6native8rand_outEN3c108ArrayRefIlEESt8optionalINS_9GeneratorEERNS_6TensorE': [
        ('rand.out', 'CompositeExplicitAutograd', 'at::native::rand_out(c10::ArrayRef<long>, std::optional<at::Generator>, at::Tensor&)'),
    ],
    '_ZN2at6native8var_meanERKNS_6TensorEN3c1016OptionalArrayRefIlEERKSt8optionalINS4_6ScalarEEb': [
        ('var_mean.correction', 'CPU', 'at::native::var_mean(at::Tensor const&, c10::OptionalArrayRef<long>, std::optional<c10::Scalar> const&, bool)'),
    ],
    '_ZN2at6native8var_meanERKNS_6TensorEN3c1016OptionalArrayRefIlEEbb': [
        ('var_mean.correction', 'CPU', 'at::native::var_mean(at::Tensor const&, c10::OptionalArrayRef<long>, bool, bool)'),
    ],
    '_ZN2at6native8var_meanERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEERKSt8optionalINS4_6ScalarEEb': [
        ('var_mean.correction', 'CPU', 'at::native::var_mean(at::Tensor const&, c10::ArrayRef<at::Dimname>, std::optional<c10::Scalar> const&, bool)'),
    ],
    '_ZN2at6native8var_meanERKNS_6TensorEN3c108ArrayRefINS_7DimnameEEEbb': [
        ('var_mean.correction', 'CPU', 'at::native::var_mean(at::Tensor const&, c10::ArrayRef<at::Dimname>, bool, bool)'),
    ],
    '_ZN2at6native8var_meanERKNS_6TensorEb': [
        ('var_mean.correction', 'CPU', 'at::native::var_mean(at::Tensor const&, bool)'),
    ],
    '_ZN2at6native9index_putERKNS_6TensorERKN3c104ListISt8optionalIS1_EEES3_b': [
        ('index_put', 'CompositeExplicitAutograd', 'at::native::index_put(at::Tensor const&, c10::List<std::optional<at::Tensor> > const&, at::Tensor const&, bool)'),
    ],
    '_ZN2at6native9logit_outERKNS_6TensorESt8optionalIdERS1_': [
        ('logit.out', 'CPU', 'at::native::logit_out(at::Tensor const&, std::optional<double>, at::Tensor&)'),
    ],
    '_ZN2at6native9remainderERKN3c106ScalarERKNS_6TensorE': [
        ('remainder.Scalar', 'CompositeExplicitAutograd', 'at::native::remainder(c10::Scalar const&, at::Tensor const&)'),
    ],
    '_ZN2at6native9remainderERKNS_6TensorERKN3c106ScalarE': [
        ('remainder.Scalar', 'CompositeExplicitAutograd', 'at::native::remainder(at::Tensor const&, c10::Scalar const&)'),
    ],
    '_ZN2at6native9stack_outEN3c108ArrayRefINS_6TensorEEElRS3_': [
        ('stack.out', 'CompositeExplicitAutograd', 'at::native::stack_out(c10::ArrayRef<at::Tensor>, long, at::Tensor&)'),
    ],
    '_ZN2at6native9zeros_outEN3c108ArrayRefIlEERNS_6TensorE': [
        ('zeros.out', 'CompositeExplicitAutograd', 'at::native::zeros_out(c10::ArrayRef<long>, at::Tensor&)'),
    ],
}
