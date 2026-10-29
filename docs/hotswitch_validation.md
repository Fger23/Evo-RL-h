# Hot-switch implementation validation — 2026-10-29

This change is implemented against Fger23/Evo-RL commit
`e9d277ae0650fb3fb7e991fcfa87399b20ce1a66` and published to
https://github.com/Fger23/Evo-RL-h. See `hotswitch.md` for operating commands.

## Executed validation

- Offline tests: **94 passed, 2 skipped, 0 failed**. The two skipped tests require
  a real local ACP dataset which was not available; network access was disabled
  for these tests. The JUnit XML is included below.
- Real localhost gRPC with simulated policies and simulated control acceptance:
  ordinary async and Training-Time RTC, shared and dual resident routes, **200
  switches per group**, 1,600 accepted simulated commands per group.
- The normal benchmark reported zero RPC errors, rejected responses, stale
  response/error deliveries, late chunks, and queue underruns. Separate tests
  exercised failure injection, atomicity, rollback, and identity isolation.
- Training-Time RTC simulation used zero-prefix bootstrap after each task switch
  and prefix length 3 for subsequent refills. Raw and processed actions remained
  paired; accepted-command counts determined response alignment.
- A continuously running offline client accepted A→B→A through its local control
  mailbox, with monotonically increasing epochs and 500 simulated commands.
- New modules passed Ruff and Python compilation. Protocol bindings were generated
  with grpcio-tools 1.73.1 from the checked-in proto source.

## Raw data

`benchmarks/hotswitch/2026-10-29-simulated/` contains `tests.xml` and two benchmark
folders. Each includes summary JSON plus per-switch, RPC and simulated control
acceptance JSONL for both shared and dual routes. Durations use local monotonic
clocks; server lock wait and inference durations are recorded separately. P99
estimates from 200 samples have limited precision.

These results measure simulated policies and an in-memory control interface.
They do not establish real PI05 latency, GPU memory savings, physical robot
latency, or task success. GPU memory and task success fields remain null.

## Not executed

- Real PI05 joint LoRA injection, short training, save/restore and A→B→A validation.
- Real GPU model loading, warmup, output equivalence and performance comparison.
- Formal task A/B ACP plus Training-Time RTC training.
- Hardware connection, calibration, robot commands and rollout success evaluation.

The implementation workstation had CPU-only PyTorch, incompatible stock
Transformers, no complete PI05 weight snapshot, and no confirmed A/B datasets.
Its physical GPU was occupied during the initial check. Model-dependent commands
reject missing prerequisites; no training result or task adapter is fabricated.
The supplied configuration files retain null values for resources not confirmed.

Uploading this source snapshot to the server does not by itself validate that
server's model environment or start any process. GPU training and hardware
validation remain separate steps.
