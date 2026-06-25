"""mobile - ZeroMQ-brokered ExecuTorch differential fuzzing.

Producers generate graphs and lower each to an ExecuTorch .pte (with the eager
reference); a broker streams jobs to executor clients (local now, a mobile phone
later) that run the .pte on the ExecuTorch runtime; the broker diffs eager vs
ExecuTorch and tallies. See README.md for the architecture and run commands.
"""
