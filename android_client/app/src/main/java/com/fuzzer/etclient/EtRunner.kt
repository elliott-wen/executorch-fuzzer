package com.fuzzer.etclient

import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import java.io.File

/**
 * Loads a .pte into the ExecuTorch runtime and runs forward(inputs). Mirrors
 * mobile/executor/runtime_run.py — the work a phone client does natively.
 *
 * A hard kernel failure is a NATIVE abort that takes the whole process down; the crash
 * breadcrumb in FuzzClient then reports the killed job as CRASH — the fuzzing signal we want.
 * scratchDir is PER-WORKER (see WorkerService) so job.pte never collides across processes.
 */
class EtRunner(private val scratchDir: File) {

    /** Write the .pte bytes to a file (Module.load needs a path), run, serialize raw outputs.
     *  The broker diffs these vs eager — the phone does no comparison. */
    fun run(pte: ByteArray, inputs: List<EValue>): List<Protocol.RawOut> {
        val f = File(scratchDir, "job.pte")
        f.writeBytes(pte)
        val module = Module.load(f.absolutePath)
        try {
            val outs: Array<EValue> = module.forward(*inputs.toTypedArray())
            return outs.filter { it.isTensor }.map { Tensors.etToRaw(it.toTensor()) }
        } finally {
            module.destroy()
        }
    }
}
