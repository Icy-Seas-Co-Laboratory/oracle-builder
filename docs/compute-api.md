# Retired compute API

The push-based `/compute/*` protocol and `oracle-serve` process are not part of
Oracle Builder's supported runtime. They are retained only in source history
to make one-way migration and forensic reading possible; do not deploy or call
them.

New execution uses the [Oracle worker](oracle-worker.md) pull protocol. The
Orchestrator creates a portable WorkUnit, grants its artifacts to an admitted
worker, and is the only service allowed to publish resulting outputs.
