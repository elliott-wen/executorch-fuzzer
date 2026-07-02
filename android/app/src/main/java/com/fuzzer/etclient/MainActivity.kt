package com.fuzzer.etclient

import android.content.Context
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Coordinator (runs in the main process). Starts K [Lane]s, each a stable ZMQ connection to the
 * broker that delegates execution to a disposable [ExecutorService] in its own OS process. The
 * lanes never touch the ExecuTorch runtime, so a graph that aborts the runtime kills only an
 * executor process — the lane records CRASH and respawns it, with no broker churn.
 *
 * The lanes run as threads in this Activity's process, so keep the app foregrounded for a session
 * (FLAG_KEEP_SCREEN_ON holds it alive). Per-lane progress is in logcat (ETFuzz-ExecutorN) and here.
 */
class MainActivity : AppCompatActivity() {

    private val lanes = ArrayList<Lane>()
    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        title = "ETClient $BUILD_TAG"

        val hostField = findViewById<EditText>(R.id.host)
        val portField = findViewById<EditText>(R.id.port)
        val countField = findViewById<EditText>(R.id.workers)
        val timeoutField = findViewById<EditText>(R.id.timeout)
        val logView = findViewById<TextView>(R.id.log)
        val logScroll = findViewById<ScrollView>(R.id.logScroll)
        val button = findViewById<Button>(R.id.startBtn)

        // Remember the last-used settings across launches (saved on Start, below).
        val prefs = getSharedPreferences("etfuzz", Context.MODE_PRIVATE)
        prefs.getString("host", null)?.let { hostField.setText(it) }
        prefs.getString("port", null)?.let { portField.setText(it) }
        prefs.getString("workers", null)?.let { countField.setText(it) }
        prefs.getString("timeout", null)?.let { timeoutField.setText(it) }

        // Bounded, auto-scrolling log — a TextView appended-to forever would balloon and lag/OOM.
        val logLines = ArrayDeque<String>()
        fun log(m: String) = runOnUiThread {
            logLines.addLast(m)
            while (logLines.size > MAX_LOG_LINES) logLines.removeFirst()
            logView.text = logLines.joinToString("\n")
            logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }   // auto-scroll to the latest line
        }
        log("=== $BUILD_TAG ===")   // confirm which build is running

        button.setOnClickListener {
            if (!running) {
                val host = hostField.text.toString().trim()
                val port = portField.text.toString().toIntOrNull() ?: 15555
                val n = (countField.text.toString().toIntOrNull() ?: 4).coerceIn(1, Executors.MAX)
                val timeoutMs = (timeoutField.text.toString().toIntOrNull() ?: 30).toLong() * 1000
                prefs.edit()
                    .putString("host", host)
                    .putString("port", portField.text.toString())
                    .putString("workers", countField.text.toString())
                    .putString("timeout", timeoutField.text.toString())
                    .apply()
                running = true
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                for (i in 0 until n) {
                    Lane(i, host, port, port + 1, this, Executors.CLASSES[i], timeoutMs, ::log)
                        .also { lanes.add(it); it.start() }
                }
                button.text = "Stop"
                log("coordinator: started $n lanes → $host:$port (ctrl ${port + 1})")
            } else {
                stopAll(::log)
                button.text = "Start"
            }
        }
    }

    private fun stopAll(log: (String) -> Unit) {
        running = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        for (lane in lanes) lane.stop()
        lanes.clear()
        log("coordinator: stopped")
    }

    override fun onDestroy() {
        if (running) stopAll {}
        super.onDestroy()
    }

    companion object {
        private const val MAX_LOG_LINES = 500
        // Bump this every rebuild so the running build is unmistakable in the title bar + log.
        const val BUILD_TAG = "build-6 executor-recycle/25"
    }
}
