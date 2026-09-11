package com.fuzzer.etclient

import org.json.JSONArray
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream

/**
 * Kotlin port of mobile/net/protocol.py (the parts the client needs).
 *
 * INVERTED diff: the broker keeps the eager reference and does the comparison, so the phone
 * only ever sees the lean JOB and returns its RAW outputs — it never decodes eager and never
 * compares (no Comparator, no dtype-matching for the diff).
 *
 *   job    (broker → phone): frame[0]=JSON header {job_id, inputs:[{dtype,dims}]}
 *                            frame[1]=gzip(pte) ; frame[2..]=gzip(input raw LE bytes)
 *   result (phone → broker): frame[0]=JSON header {job_id, status, detail, outputs:[{dtype,dims}]}
 *                            frame[1..]=gzip(output raw LE bytes)   (only when status=="RAN")
 *
 * `dtype` is the torch ScalarType integer code — identical to ExecuTorch's DType.jniCode.
 */
object Protocol {

    /** A serialized ExecuTorch OUTPUT tensor as it travels back on the wire. */
    class RawOut(val dtypeCode: Int, val dims: LongArray, val raw: ByteArray)

    class Job(
        val jobId: String,
        val pte: ByteArray,
        val inputs: List<EValue>,
    )

    private fun gunzip(b: ByteArray): ByteArray =
        GZIPInputStream(b.inputStream()).use { it.readBytes() }

    private fun gzip(b: ByteArray): ByteArray {
        val bos = ByteArrayOutputStream(b.size)
        GZIPOutputStream(bos).use { it.write(b) }
        return bos.toByteArray()
    }

    private fun meta(arr: JSONArray, i: Int): Pair<Int, LongArray> {
        val o = arr.getJSONObject(i)
        val dimsJson = o.getJSONArray("dims")
        val dims = LongArray(dimsJson.length()) { dimsJson.getLong(it) }
        return o.getInt("dtype") to dims
    }

    /** Broker → phone: decode the lean job → (job_id, pte, input EValues). No eager. */
    fun decodeJob(frames: List<ByteArray>): Job {
        val header = JSONObject(String(frames[0], Charsets.UTF_8))
        val payload = ArrayList<ByteArray>(frames.size - 1)
        for (i in 1 until frames.size) payload.add(gunzip(frames[i]))

        val inMetas = header.getJSONArray("inputs")
        val nIn = inMetas.length()
        val pte = payload[0]
        val inputs = ArrayList<EValue>(nIn)
        for (k in 0 until nIn) {
            val (code, dims) = meta(inMetas, k)
            inputs.add(EValue.from(tensorFromRaw(code, dims, payload[1 + k])))
        }
        return Job(header.getString("job_id"), pte, inputs)
    }

    /** Build an ExecuTorch input Tensor from raw little-endian bytes + dtype code. */
    private fun tensorFromRaw(code: Int, dims: LongArray, raw: ByteArray): Tensor {
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        return when (code) {
            0 -> Tensor.fromBlobUnsigned(raw, dims)                       // uint8
            1 -> Tensor.fromBlob(raw, dims)                              // int8
            2 -> Tensor.fromBlobInt16(ShortArray(raw.size / 2) { bb.short }, dims)   // int16
            3 -> Tensor.fromBlob(IntArray(raw.size / 4) { bb.int }, dims)        // int32
            4 -> Tensor.fromBlob(LongArray(raw.size / 8) { bb.long }, dims)      // int64
            5 -> Tensor.fromBlob(ShortArray(raw.size / 2) { bb.short }, dims)    // float16 (bit pattern)
            6 -> Tensor.fromBlob(FloatArray(raw.size / 4) { bb.float }, dims)    // float32
            7 -> Tensor.fromBlob(DoubleArray(raw.size / 8) { bb.double }, dims)  // float64
            11 -> Tensor.fromBlobBool(raw, dims)                         // bool (1 byte/elem, 0/1)
            15 -> Tensor.fromBlobBFloat16(ShortArray(raw.size / 2) { bb.short }, dims) // bfloat16 (bit pattern)
            else -> throw IllegalArgumentException("unsupported input dtype code $code on Android")
        }
    }

    /**
     * Phone → broker: encode the result. On "RAN", `outputs` carries the raw ET output tensors
     * (the broker diffs them vs eager). On "SKIP"/"CRASH"/"TIMEOUT", pass null — no tensors.
     */
    fun encodeResult(jobId: String, status: String, detail: String,
                     outputs: List<RawOut>?): List<ByteArray> {
        val metas = JSONArray()
        val frames = ArrayList<ByteArray>()
        if (outputs != null) {
            for (o in outputs) {
                val m = JSONObject()
                m.put("dtype", o.dtypeCode)
                val dims = JSONArray()
                for (d in o.dims) dims.put(d)
                m.put("dims", dims)
                metas.put(m)
                frames.add(gzip(o.raw))
            }
        }
        val h = JSONObject()
        h.put("job_id", jobId)
        h.put("status", status)
        h.put("detail", detail)
        h.put("outputs", metas)
        val out = ArrayList<ByteArray>(frames.size + 1)
        out.add(h.toString().toByteArray(Charsets.UTF_8))
        out.addAll(frames)
        return out
    }
}
