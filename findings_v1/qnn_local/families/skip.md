# SKIP family — QNN can't run the graph (17,989 jobs, 47%)

Stage recovered by re-running a 400-job sample through the runner with `run.log` preserved
(`skip_stage.py`). Each cause deep-dived to the exact triggering tensor/op.
Raw: `../localize/skip_stage.tsv`.

| Stage | sample % | est. jobs | Repro |
|---|---:|---:|---|
| alloc_output | 49.5% | ~8,900 | `w0:1092` |
| execute (6004) | 30.8% | ~5,540 | `w0:2046` |
| alloc_input | 15.0% | ~2,700 | `w1:974` |
| load_method | 4.8% | ~860 | `w10:827` |

---

## BUG 7 — 0-element tensors can't be allocated  (alloc_input + alloc_output, ~64% of skips)

The runner sizes a `CustomMemory` buffer to `tensor->nbytes()`; a tensor with any 0-sized dim
has `nbytes()==0` and `Allocate(0)` returns false. Confirmed empties:

- **`w0:1092`** output `out[7]` shape **`(0,)`** — produced upstream by a slice/`unbind`/`amin` chain.
- **`w1:974`** output `out[6]` shape **`(0, 0)`** from **`broadcast_to(n15, [0, 0])`** — an
  explicit 0-extent broadcast.

Mechanism: the generator emits 0-extent shapes (`broadcast_to([0,…])`, slice/narrow to length 0,
`unbind` of a 0-dim), and neither the runner nor the HTP allocator handles a 0-byte tensor.
This is a **runner/QNN limitation with empty tensors**, dtype-independent — the biggest single
skip driver.

---

## BUG 8 — Rank/batch mismatch → `qnn_graph_execute` Error 6004  (execute, ~31% of skips)

`6004 = QNN_GRAPH_ERROR_INVALID_TENSOR` (`QNN_MIN_ERROR_GRAPH 6000 + 4`). The HTP requires the
leading ("batch") dim consistent across an op's I/O; rank-changing ops break it.

- **`w0:2046`**: chain includes **`mean.out(n2, None, keepdim=True)`** (reduce-all → shape
  `(1,1,1,1)`) and **`broadcast_to(n6, [3])`** (rank 0/1 → 1). The runtime warns
  `Tensor rank mismatches: 2 vs 1, ignore current operation`, then
  `Mismatching input and output batch sizes` → 6004.

Mechanism: same root as the rank-altering CRASH/MISMATCH cases — `view`→`(N,1)`, broadcast of
rank-2 vs rank-1, reductions/`select` to scalar. Compiles (partition accepts), fails at execute.

---

## BUG 9 — Graph won't finalize on HTP  (load_method, ~5% of skips)

`load_program(...).load_method("forward")` returns non-Ok (`w10:827`): the HTP graph preparer
rejects the context binary before any execution. Rarer; usually an op/shape the partitioner
admitted but the HTP graph builder later refuses.

## Reproduce
```bash
source android-dev/android-env.sh
export PYTHONPATH=/data/jwen929 LD_LIBRARY_PATH="$PWD/pytorch_ref/executorch/build-x86/lib:$LD_LIBRARY_PATH"
echo "w0:2046" > /tmp/one.txt
.venv/bin/python findings_v1/qnn_local/skip_stage.py /tmp/one.txt 1   # -> stage + qnn error
```
