# mobile

A differential fuzzer for ExecuTorch. It builds random PyTorch graphs, records what
PyTorch says they produce, compiles them for a mobile/embedded backend, runs them
there, and compares. Anything that disagrees is a bug in the compile-or-run stack.

Running it is three stages, always in this order:

| stage | what you get | where it lands |
|---|---|---|
| 1. **generate the oracle** | random graphs + the PyTorch answer for each | an *oracle corpus* directory |
| 2. **lower the graphs** | one `.pte` per graph, compiled for one backend | a *pte corpus* directory |
| 3. **execute the graphs** | a verdict per graph: OK / MISMATCH / CRASH / TIMEOUT / SKIP | a skip-log `.tsv` |

The oracle corpus is the expensive part and it keeps: once generated, you can lower and
execute it for as many backends as you like. It is generated *with* one backend in mind
(the generator solves that backend's extra constraints too, so fewer graphs get rejected
later), but nothing stops you pointing another backend at it.

Every command below is run **from inside this folder**:

```bash
cd /data/jwen929/mobile
export PYTHONPATH=/data/jwen929      # the scripts set this themselves
```

## The short version

```bash
scripts/run.sh xnnpack
```

That runs all three stages for the `xnnpack` backend with 10,000 single-op graphs and
prints where everything went. Add a tag, a graph count, and a graph size if you want
something other than the defaults:

```bash
scripts/run.sh vulkan v2 10000 8      # tag "v2", 10k graphs, 8 ops each
```

**Graph size matters.** `1` (the default) means one operator per graph, so every
failure names exactly one op and needs no detective work. `8` gives multi-op graphs,
which find bugs that only appear when operators are composed — at the cost of having
to narrow down which op was at fault.

## Running the stages one at a time

`run.sh` just calls three scripts in `scripts/`. Use them directly when you want to
repeat a stage without redoing the ones before it — re-lowering an unchanged oracle
corpus is the common case.

```bash
# 1. generate the oracle: <backend> <out-dir> [count] [nodes]
scripts/gen.sh xnnpack corpus/oracle_xnn 10000 1

# 2. lower: <backend> <oracle-dir> <out-dir> [workers]
scripts/lower.sh xnnpack corpus/oracle_xnn corpus/pte_xnn

# 3. execute: <backend> <pte-dir> <skip-log> [clients]
scripts/execute.sh xnnpack corpus/pte_xnn tmp/results_xnn.tsv
```

All three are resumable: point stage 1 at an existing oracle directory and it adds to
it; point stage 2 at a partly-lowered output and it continues.

Stage 3 exits non-zero when it found failures. That is the tool reporting findings,
not the script breaking.

### Useful knobs

```bash
QUANTIZE=1 scripts/lower.sh openvino corpus/oracle_ov corpus/pte_ov   # int8 path
COUNT=200  scripts/lower.sh cuda corpus/oracle_cu corpus/pte_cu       # lower a probe batch only
scripts/execute.sh xnnpack corpus/pte_xnn tmp/r.tsv 32               # 32 parallel clients
```

## Backends

```
portable  xnnpack  vulkan  vgf  openvino  cadence  nxp  qualcomm
ethos-u   cortex-m  cuda    coreml  mps    mediatek  samsung
```

Stages 1 and 2 work for all of them. Stage 3 needs something that can actually run the
program, and that is where the backends differ:

* **Runs on this machine** — portable, xnnpack, vulkan, vgf, openvino, cadence, nxp,
  qualcomm, ethos-u, cuda. A host runtime, simulator, or emulator stands in for the
  real chip, so `scripts/execute.sh` handles these on its own.
* **Needs a physical device** — vulkan and qualcomm on a real phone (the host paths
  above are emulation, and they do not always agree), plus samsung (an Exynos phone)
  and mediatek (a MediaTek one), which have no host path at all.
* **Needs other hardware we don't have here** — coreml and mps (a Mac), cortex-m (a
  Cortex-M board).

For anything in the last two groups, `execute.sh` stops with an explanation instead of
failing oddly, so `run.sh` is safe to point at any backend — you still get stages 1 and
2 and the lowering results.

A few backends execute through a native runner binary that has to be built once:

```bash
local_client/vulkan_client/build_runner.sh     # likewise vgf_client, cadence_client, nxp_client
```

`scripts/env.sh` holds the per-backend environment — SDK paths, which Python to use,
library workarounds. It is sourced by the other scripts; you never run it yourself, and
you shouldn't need to read it unless you're adding a backend.

## Running on a phone

Stages 1 and 2 are unchanged — you still generate and lower on this machine. Only stage 3
moves: instead of a host client, the phone runs `android_client/`, an Android app that
connects to the same broker and pulls jobs over the network. The feeder can't tell a phone
from a local client, so results, skip-logs, and replay all work the same way.

```bash
cd android_client
./build_et_aar.sh                 # once: build the ExecuTorch runtime for Android (~40 min)
./gradlew assembleDebug           # the app itself
adb install -r app/build/outputs/apk/debug/app-debug.apk
cd ..
```

Then start a broker on this machine, open the app on the phone, type in the broker's IP
and port (default `15555`) and how many parallel workers you want, press Start, and feed
the corpus:

```bash
python -m executor.broker                                  # here
python -m executor.feed --corpus corpus/pte_vulkan --skip-log tmp/phone.tsv
```

Keep the app in the foreground for the whole session. Each worker runs graphs in its own
process, so a graph that takes down the runtime costs you that one job (recorded as a
CRASH) rather than the run.

`./rebuild_aar_fast.sh` re-does only the changed native sources after the first full build.
The toolchain (NDK, JDK, SDK, vendor SDKs) lives in `android-dev/`.

## Reading the results

Stage 3 prints a running tally and a final count:

```
[t=  42s] 8000/10000 (190.5/s)  ok=7912 mism=61 crash=9 timeout=0 skip=18
```

* **OK** — the backend agreed with PyTorch.
* **MISMATCH** — it ran and produced different numbers. This is the interesting one.
* **CRASH / TIMEOUT** — it fell over or hung.
* **SKIP** — the graph never made it that far (an unsupported op, say). Expected in
  quantity; the skip-log groups them by reason.

Every failing row goes into the skip-log TSV you named, tagged with the graph's *token* —
the id that identifies one graph everywhere in the pipeline. Feed a token back to see what
happened:

```bash
TOK=060fbd4325434e6086723491628ef7bc

# print the graph's source — what PyTorch was asked to compute
python -m generator.oracle corpus/oracle_xnn --show $TOK

# re-run just that one graph, with full per-output detail as JSON
python -m executor.feed --corpus corpus/pte_xnn $TOK
```

Re-running one job needs a broker and a client alive — easiest is to leave a
`scripts/execute.sh` fleet running in another terminal, or start them by hand
(`python -m executor.broker`, then the client listed in `env.sh` for your backend).

You can also replay a whole class of failures at once:

```bash
python -m executor.feed --from-tsv tmp/results_xnn.tsv --status MISMATCH -n 20 \
    --corpus corpus/pte_xnn
```

## Running the stages by hand

The scripts are wrappers over two generator commands and the executor. If you'd rather
drive them directly:

```bash
export MOBILE_BACKENDS=xnnpack       # plus PYTHONPATH, above

python -m generator.oracle corpus/oracle_xnn --count 10000 --nodes 1 --target xnnpack
python -m generator.lower  corpus/oracle_xnn corpus/pte_xnn --backend xnnpack

python -m executor.broker &                                    # one, routes work
python local_client/xnnpack_client/xnnpack_client.py --host 127.0.0.1 &   # one or more
python -m executor.feed --corpus corpus/pte_xnn --skip-log tmp/results.tsv
```

Add `--stats` to either generator command to report on a corpus without adding to it.

Clients don't have to be local — another machine, or a phone running `android_client/`,
can pull from the same broker.

## More

* `scripts/README.md` — the stage scripts and the environment traps they encode.
* `local_client/README.md` — what each execution client is and what it needs.
* `android_client/` — the phone app used as an execution client.
* `docs/findings/` — results from previous runs, per backend.

How the pipeline works internally is documented in the code: each module's docstring is
the source of truth, starting with `generator/oracle/`, `generator/lower/`, and
`executor/`.
