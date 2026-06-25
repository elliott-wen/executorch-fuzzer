package com.fuzzer.etclient

import org.pytorch.executorch.DType
import org.pytorch.executorch.Tensor
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Serializes an ExecuTorch OUTPUT tensor to raw little-endian bytes for the wire. Since the
 * broker now does the diff, the phone never decodes eager or compares — it just ships whatever
 * the runtime produced, in the tensor's NATIVE dtype (fp16/bf16 go as their raw bit pattern, no
 * lossy float round-trip), and the broker reconstructs it exactly via torch.frombuffer.
 */
object Tensors {

    /** ExecuTorch Tensor → (dtype code + dims + raw LE bytes). */
    fun etToRaw(t: Tensor): Protocol.RawOut {
        val (code, raw) = when (t.dtype()) {
            DType.FLOAT -> 6 to floatLE(t.dataAsFloatArray)
            DType.DOUBLE -> 7 to doubleLE(t.dataAsDoubleArray)
            DType.INT64 -> 4 to longLE(t.dataAsLongArray)
            DType.INT32 -> 3 to intLE(t.dataAsIntArray)
            DType.INT16 -> 2 to shortLE(t.dataAsShortArray)
            DType.HALF -> 5 to shortLE(t.dataAsShortArray)        // raw fp16 bit pattern
            DType.BFLOAT16 -> 15 to shortLE(t.dataAsShortArray)   // raw bf16 bit pattern
            DType.INT8 -> 1 to t.dataAsByteArray
            DType.UINT8 -> 0 to t.dataAsUnsignedByteArray
            DType.BOOL -> 11 to t.dataAsByteArray                 // 0/1 bytes
            else -> throw IllegalArgumentException("serialize: dtype ${t.dtype()} unsupported on Android")
        }
        return Protocol.RawOut(code, t.shape(), raw)
    }

    private fun floatLE(a: FloatArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (x in a) bb.putFloat(x)
        return bb.array()
    }

    private fun doubleLE(a: DoubleArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 8).order(ByteOrder.LITTLE_ENDIAN)
        for (x in a) bb.putDouble(x)
        return bb.array()
    }

    private fun longLE(a: LongArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 8).order(ByteOrder.LITTLE_ENDIAN)
        for (x in a) bb.putLong(x)
        return bb.array()
    }

    private fun intLE(a: IntArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (x in a) bb.putInt(x)
        return bb.array()
    }

    private fun shortLE(a: ShortArray): ByteArray {
        val bb = ByteBuffer.allocate(a.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (x in a) bb.putShort(x)
        return bb.array()
    }
}
