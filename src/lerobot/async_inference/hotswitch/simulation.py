"""Hardware-free policy and control doubles for protocol tests and timings."""

import time


class SimulatedBackend:
    def __init__(self, mode="shared", delay_s=0.0001, rtc=True):
        if mode not in ("shared", "dual"):
            raise ValueError(mode)
        self.task_ids = ("A", "B")
        self.mode = mode
        self.delay_s = delay_s
        self.active = None
        self.warmed = set()
        self.base_load_count = 1 if mode == "shared" else 2
        self.adapter_load_count = 2 if mode == "shared" else 0
        self.task_metadata = {key: {"rtc_mode": "training_time" if rtc else "async",
            "prefix_steps": 2 if rtc else 0, "max_prefix_steps": 3 if rtc else 0,
            "action_dim": 2, "chunk_size": 8} for key in self.task_ids}

    def activate(self, task_id):
        if task_id not in self.task_ids:
            raise ValueError(task_id)
        self.active = task_id

    def reset(self, task_id):
        pass

    def warmup(self, task_id):
        self.activate(task_id)
        self.infer(task_id, {})
        if self.task_metadata[task_id]["prefix_steps"]:
            self.infer(task_id, {}, [[0.0, 0.0], [0.1, 0.1]], 2)
        self.warmed.add(task_id)

    def infer(self, task_id, observation, action_prefix=None, prefix_steps=0):
        if self.active != task_id:
            raise RuntimeError("Adapter and task were mixed")
        time.sleep(self.delay_s)
        bias = 0.2 if task_id == "A" else -0.2
        raw = [[bias + i * 0.01, bias - i * 0.01] for i in range(8)]
        if prefix_steps:
            raw[:prefix_steps] = [list(row) for row in action_prefix]
        scale = 2.0 if task_id == "A" else 3.0
        processed = [[x * scale for x in row] for row in raw]
        return raw, processed

    def offline_observation(self, task_id):
        return {"offline": True}


class SimulatedRobot:
    """This object cannot discover ports, connect hardware, or send physical commands."""

    def __init__(self):
        self.holds = 0
        self.accepted = []

    def hold(self):
        self.holds += 1

    def send_action(self, action):
        action = list(action)
        self.accepted.append(action)
        return action
