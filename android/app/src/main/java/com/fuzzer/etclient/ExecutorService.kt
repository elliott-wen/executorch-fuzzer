package com.fuzzer.etclient

import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.util.Timer
import java.util.TimerTask

/** Lane↔executor IPC constants. The bulky JOB is spooled to scratch/job.bin; the small RESULT
 *  rides back in the MSG_DONE Bundle (K_RESULT = packed result frames), no file. */
object Ipc {
    const val MSG_RUN = 1     // coordinator → executor: run the job spooled to scratch/job.bin
    const val MSG_DONE = 2    // executor → coordinator: arg1=token, data={status, result?}
    const val K_STATUS = "status"
    const val K_RESULT = "result"   // FrameFile.pack(result frames); absent on TIMEOUT
}

/**
 * A DISPOSABLE executor process. It does ONLY the crash-prone work — decode the spooled job, run
 * the .pte on the ExecuTorch runtime, and serialize the RAW (gzipped) outputs back in the reply
 * Bundle (the broker does the diff). It holds NO broker connection, so when a graph aborts the
 * runtime (native SIGABRT) only THIS process dies; the coordinator (Lane, a stable process) sees the
 * binding drop and records CRASH. A hung kernel is caught by the self-watchdog, which replies TIMEOUT
 * then kills this process.
 *
 * Each subclass (Executor0..7) is pinned to its own OS process via android:process in the manifest.
 * scratch = cacheDir/<ExecutorN>/ is per-process, so job.bin / job.pte never collide across workers.
 */
open class ExecutorService : Service() {
    private lateinit var thread: HandlerThread
    private lateinit var messenger: Messenger
    private val scratch by lazy { File(cacheDir, javaClass.simpleName).apply { mkdirs() } }
    private val runner by lazy { EtRunner(scratch) }
    private val tag get() = "ETFuzz-${javaClass.simpleName}"

    override fun onCreate() {
        super.onCreate()
        thread = HandlerThread("exec").apply { start() }
        messenger = Messenger(object : Handler(thread.looper) {
            override fun handleMessage(msg: Message) {
                if (msg.what == Ipc.MSG_RUN) runJob(msg.arg1, msg.arg2.toLong(), msg.replyTo)
            }
        })
    }

    private fun reply(to: Messenger?, token: Int, status: String, result: ByteArray?) {
        try {
            to?.send(Message.obtain(null, Ipc.MSG_DONE, token, 0).apply {
                data = Bundle().apply {
                    putString(Ipc.K_STATUS, status)
                    if (result != null) putByteArray(Ipc.K_RESULT, result)  // packed, gzipped outputs
                }
            })
        } catch (_: Throwable) { /* coordinator gone; nothing to do */ }
    }

    private fun runJob(token: Int, timeoutMs: Long, replyTo: Messenger?) {
        // Self-watchdog: a hung Module.forward never returns, so a separate thread reports TIMEOUT
        // and kills this process. (A native abort needs no watchdog — it kills us outright and the
        // coordinator infers CRASH from the dropped binding.)
        val watchdog = Timer(true)
        watchdog.schedule(object : TimerTask() {
            override fun run() {
                reply(replyTo, token, "TIMEOUT", null)   // no payload — the lane builds the TIMEOUT result
                Log.w(tag, "watchdog: job exceeded timeout — killing executor")
                Process.killProcess(Process.myPid())
            }
        }, if (timeoutMs > 0) timeoutMs else 30_000L)

        val frames = try {
            FrameFile.read(File(scratch, "job.bin"))
        } catch (e: Throwable) {
            watchdog.cancel()
            reply(replyTo, token, "SKIP", FrameFile.pack(Protocol.encodeResult("?", "SKIP",
                "executor read ${e.javaClass.simpleName}: ${e.message}", null)))
            return
        }
        // Recover job_id even if full decode later fails, so the broker can still attribute the row.
        val jobId = try {
            JSONObject(String(frames[0], Charsets.UTF_8)).getString("job_id")
        } catch (_: Throwable) { "?" }
        try {
            val job = Protocol.decodeJob(frames)
            val outs = runner.run(job.pte, job.inputs)      // run + serialize raw outputs (gzipped in encodeResult)
            watchdog.cancel()
            reply(replyTo, token, "RAN", FrameFile.pack(Protocol.encodeResult(jobId, "RAN", "", outs)))
        } catch (e: Throwable) {                  // Java-level error → SKIP (a native abort kills us instead)
            watchdog.cancel()
            reply(replyTo, token, "SKIP", FrameFile.pack(Protocol.encodeResult(jobId, "SKIP",
                "executor ${e.javaClass.simpleName}: ${e.message}", null)))
        }
    }

    override fun onBind(intent: Intent?): IBinder = messenger.binder
    override fun onDestroy() { thread.quitSafely(); super.onDestroy() }
}

// Distinct classes so the manifest can give each its own android:process (its own OS process).
class Executor0 : ExecutorService()
class Executor1 : ExecutorService()
class Executor2 : ExecutorService()
class Executor3 : ExecutorService()
class Executor4 : ExecutorService()
class Executor5 : ExecutorService()
class Executor6 : ExecutorService()
class Executor7 : ExecutorService()
class Executor8 : ExecutorService()
class Executor9 : ExecutorService()
class Executor10 : ExecutorService()
class Executor11 : ExecutorService()
class Executor12 : ExecutorService()
class Executor13 : ExecutorService()
class Executor14 : ExecutorService()
class Executor15 : ExecutorService()

object Executors {
    val CLASSES = listOf(
        Executor0::class.java, Executor1::class.java, Executor2::class.java, Executor3::class.java,
        Executor4::class.java, Executor5::class.java, Executor6::class.java, Executor7::class.java,
        Executor8::class.java, Executor9::class.java, Executor10::class.java, Executor11::class.java,
        Executor12::class.java, Executor13::class.java, Executor14::class.java, Executor15::class.java,
    )
    const val MAX = 16
}
