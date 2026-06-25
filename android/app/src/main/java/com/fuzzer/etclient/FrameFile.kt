package com.fuzzer.etclient

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/**
 * Length-prefixed (de)serialization of ZeroMQ frames for the coordinator (Lane) ↔ executor
 * (ExecutorService) handoff. Format: [nframes:i32]( [len:i32][bytes] )*.
 *
 * The inbound JOB (bulky — carries the .pte) is spooled to a file (write/read): 16 lanes pushing
 * ptes through Binder at once could exceed its ~1MB transaction buffer. The outbound RESULT is a
 * few small gzipped output tensors, so it rides the Messenger Bundle in memory (pack/unpack) — no
 * disk on the hot return path.
 */
object FrameFile {
    fun write(f: File, frames: List<ByteArray>) =
        DataOutputStream(f.outputStream().buffered()).use { writeFrames(it, frames) }

    fun read(f: File): List<ByteArray> =
        DataInputStream(f.inputStream().buffered()).use { readFrames(it) }

    /** In-memory variant of [write] — for passing a (small) result through an IPC Bundle. */
    fun pack(frames: List<ByteArray>): ByteArray {
        val bos = ByteArrayOutputStream()
        DataOutputStream(bos).use { writeFrames(it, frames) }
        return bos.toByteArray()
    }

    /** In-memory variant of [read]. */
    fun unpack(blob: ByteArray): List<ByteArray> =
        DataInputStream(ByteArrayInputStream(blob)).use { readFrames(it) }

    private fun writeFrames(out: DataOutputStream, frames: List<ByteArray>) {
        out.writeInt(frames.size)
        for (fr in frames) { out.writeInt(fr.size); out.write(fr) }
    }

    private fun readFrames(ins: DataInputStream): List<ByteArray> {
        val n = ins.readInt()
        return ArrayList<ByteArray>(n).apply {
            repeat(n) { val len = ins.readInt(); add(ByteArray(len).also { ins.readFully(it) }) }
        }
    }
}
