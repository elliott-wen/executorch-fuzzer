package com.fuzzer.etclient

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import org.json.JSONObject
import org.zeromq.SocketType
import org.zeromq.ZContext
import org.zeromq.ZMQ
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * One fuzzing lane in the COORDINATOR process. The lane owns a stable ZMQ REQ connection to the
 * broker and NEVER touches the ExecuTorch runtime, so it cannot crash. It delegates the crash-prone
 * work (run .pte + diff) to a disposable [ExecutorService] in its own OS process:
 *
 *   recv pushjob ─► spool frames to executor scratch ─► MSG_RUN ─► await verdict / crash ─► RESULT
 *
 * When the executor aborts (native SIGABRT) the binding drops (onServiceDisconnected) and the lane
 * records CRASH locally — it still holds the job_id, so no breadcrumb is needed and the broker never
 * sees the connection churn. The lane then re-startService()s the executor (explicit start bypasses
 * Android's crash-loop give-up) and carries on. This mirrors the desktop client.py / runtime_run.py
 * split: coordination survives, execution is disposable.
 */
class Lane(
    private val index: Int,
    private val host: String,
    private val clientPort: Int,
    private val ctrlPort: Int,
    private val ctx: Context,
    private val execClass: Class<*>,
    private val jobTimeoutMs: Long,
    private val log: (String) -> Unit,
) {
    @Volatile private var running = false
    @Volatile private var executor: Messenger? = null     // bound executor's control channel
    @Volatile private var inFlight = false
    @Volatile private var token = 0
    @Volatile private var bound = false
    @Volatile private var recycleReq = false               // executor asked to be respawned (fd cleanup)
    private var backoffMs = 0L                             // escalating restart delay after a CRASH
    // (status, packed-result-or-null). RAN/SKIP carry the executor's packed result; CRASH/TIMEOUT null.
    private val results = LinkedBlockingQueue<Pair<String, ByteArray?>>()
    private val scratch = File(ctx.cacheDir, execClass.simpleName).apply { mkdirs() }
    private var thread: Thread? = null

    private val reply = Messenger(object : Handler(ipcLooper) {
        override fun handleMessage(msg: Message) {
            if (msg.what == Ipc.MSG_DONE && msg.arg1 == token && inFlight) {
                if (msg.data.getBoolean(Ipc.K_RECYCLE)) recycleReq = true
                results.offer((msg.data.getString(Ipc.K_STATUS) ?: "SKIP") to msg.data.getByteArray(Ipc.K_RESULT))
            }
        }
    })

    private val conn = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, b: IBinder?) { executor = Messenger(b) }
        override fun onServiceDisconnected(name: ComponentName?) {
            executor = null
            if (inFlight) results.offer("CRASH" to null)
        }
    }

    fun start() {
        running = true
        bindExecutor()
        thread = Thread { loop() }.also { it.start() }
        log("✓ lane $index up")
    }

    fun stop() {
        running = false
        if (bound) { try { ctx.unbindService(conn) } catch (_: Throwable) {}; bound = false }
        try { ctx.stopService(Intent(ctx, execClass)) } catch (_: Throwable) {}
    }

    /** (Re)create the executor with a FRESH binding. After Android's auto-restart gives up, the old
     *  binding is dead and never reconnects — only unbind+rebind forces a new onServiceConnected. */
    private fun bindExecutor() {
        if (bound) { try { ctx.unbindService(conn) } catch (_: Throwable) {}; bound = false }
        try {
            ctx.startService(Intent(ctx, execClass))                          // explicit start
            bound = ctx.bindService(Intent(ctx, execClass), conn, Context.BIND_AUTO_CREATE)
        } catch (_: Throwable) {}
    }

    /** Resurrect a crashed/never-bound executor. The backoff is the key defence: spacing restarts out
     *  keeps the crash RATE under Android's "bad process" threshold, so :eN is never blacklisted. */
    private fun ensureExecutor(): Boolean {
        if (executor != null) return true
        if (backoffMs > 0) { log("lane $index executor down — backoff ${backoffMs}ms"); Thread.sleep(backoffMs) }
        bindExecutor()
        var waited = 0
        while (running && executor == null && waited < 6000) { Thread.sleep(50); waited += 50 }
        return executor != null
    }

    private fun bumpBackoff() { backoffMs = (if (backoffMs == 0L) 500L else backoffMs * 2).coerceAtMost(30_000L) }

    private fun loop() {
        ZContext().use { zctx ->
            val req = zctx.createSocket(SocketType.REQ)
            req.heartbeatIvl = 5000; req.heartbeatTimeout = 20000; req.heartbeatTtl = 20000
            req.connect("tcp://$host:$clientPort")
            val sub = zctx.createSocket(SocketType.SUB)
            sub.connect("tcp://$host:$ctrlPort"); sub.subscribe(ByteArray(0))
            val poller = zctx.createPoller(2)
            poller.register(req, ZMQ.Poller.POLLIN); poller.register(sub, ZMQ.Poller.POLLIN)
            log("lane $index → broker tcp://$host:$clientPort")
            req.send("READY")
            var ran = 0
            while (running) {
                if (poller.poll(1000) < 0) break
                if (poller.pollin(1)) {
                    val m = sub.recv(ZMQ.DONTWAIT)
                    if (m != null && String(m) == "STOP") { log("lane $index STOP"); break }
                }
                if (!poller.pollin(0)) continue
                val frames = ArrayList<ByteArray>()
                do { frames.add(req.recv(0)) } while (req.hasReceiveMore())

                val jobId = try {
                    JSONObject(String(frames[0], Charsets.UTF_8)).getString("job_id")
                } catch (_: Throwable) { "?" }
                val (status, resultFrames) = runOnExecutor(frames, jobId)
                req.sendMore("RESULT")                       // [RESULT, header, out_raw0, out_raw1, ...]
                for (i in resultFrames.indices) {
                    if (i < resultFrames.size - 1) req.sendMore(resultFrames[i]) else req.send(resultFrames[i])
                }
                ran++
                if (status == "CRASH" || status == "TIMEOUT")
                    log("lane $index [$ran] $status $jobId")
                else if (ran % 50 == 0) log("lane $index ran $ran (last $status)")
            }
            log("lane $index stopped after $ran jobs")
        }
    }

    /** Run one job on the executor; return (statusForLog, resultFramesToSend). On RAN/SKIP the
     *  executor packs the full result (status + gzipped raw outputs) into the reply Bundle — the
     *  lane just unpacks it. On CRASH/TIMEOUT there is no reply, so the lane builds an output-less
     *  result. The bulky job (with the .pte) still spools to job.bin on the way in. */
    private fun runOnExecutor(frames: List<ByteArray>, jobId: String): Pair<String, List<ByteArray>> {
        if (!ensureExecutor()) { bumpBackoff(); return crash(jobId, "executor unavailable") }
        FrameFile.write(File(scratch, "job.bin"), frames)
        token++; results.clear(); inFlight = true
        val ex = executor
        if (ex == null) { inFlight = false; bumpBackoff(); return crash(jobId, "executor unbound") }
        try {
            ex.send(Message.obtain(null, Ipc.MSG_RUN, token, jobTimeoutMs.toInt()).apply { replyTo = reply })
        } catch (e: Throwable) {                       // DeadObjectException → executor just died
            inFlight = false; executor = null; bumpBackoff()
            return crash(jobId, "executor died on dispatch: ${e.javaClass.simpleName}")
        }
        // The executor self-kills at jobTimeoutMs; the lane backstop is a bit longer.
        val v = results.poll(jobTimeoutMs + 10_000, TimeUnit.MILLISECONDS)
        inFlight = false
        // Recycle BETWEEN jobs (inFlight now false, so the executor's exit can't be mis-read as a
        // CRASH): the executor hit its job interval and asked to be respawned to free leaked QNN fds.
        if (recycleReq && v != null && v.first != "CRASH") {
            recycleReq = false
            recycleExecutor()
        }
        return when {
            v == null -> { executor = null
                "TIMEOUT" to Protocol.encodeResult(jobId, "TIMEOUT", "no reply from executor (lane backstop)", null) }
            v.first == "CRASH" -> { executor = null; bumpBackoff()  // genuine abort → space out the restart (reason: adb logcat)
                "CRASH" to Protocol.encodeResult(jobId, "CRASH", "executor process died (native abort)", null) }
            v.first == "TIMEOUT" -> { executor = null              // clean watchdog kill — no crash penalty, no backoff
                "TIMEOUT" to Protocol.encodeResult(jobId, "TIMEOUT", "kernel exceeded watchdog (executor)", null) }
            v.second == null -> { backoffMs = 0L                   // RAN/SKIP must carry a payload; defensive
                "SKIP" to Protocol.encodeResult(jobId, "SKIP", "executor returned no result payload", null) }
            else -> { backoffMs = 0L; v.first to FrameFile.unpack(v.second!!) }   // RAN / SKIP — unpack & forward
        }
    }

    /** Cleanly retire the current executor process (frees leaked QNN/rpcmem fds) and drop the
     *  binding so the next job's ensureExecutor() spawns a FRESH one. Called between jobs only. */
    private fun recycleExecutor() {
        try { executor?.send(Message.obtain(null, Ipc.MSG_DIE)) } catch (_: Throwable) {}
        if (bound) { try { ctx.unbindService(conn) } catch (_: Throwable) {}; bound = false }
        try { ctx.stopService(Intent(ctx, execClass)) } catch (_: Throwable) {}
        executor = null
        backoffMs = 0L                    // a planned recycle is not a crash — no backoff penalty
        log("lane $index recycled executor (fd cleanup)")
    }

    private fun crash(jobId: String, detail: String): Pair<String, List<ByteArray>> =
        "CRASH" to Protocol.encodeResult(jobId, "CRASH", detail, null)

    companion object {
        private val ipcLooper by lazy { HandlerThread("lane-ipc").apply { start() }.looper }
    }
}
