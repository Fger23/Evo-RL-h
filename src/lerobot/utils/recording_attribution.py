"""Fail-closed, episode-scoped provenance for the resident recording client.

These columns describe the accepted command, not task success. The adapter ID
is the registered task routing key: the resident backend loads/activates its
PEFT adapter under exactly that key, including identical mechanism copies.
"""

from __future__ import annotations

from copy import deepcopy

FIELDS = {
    "task_id": "complementary_info.policy_task_id",
    "adapter_id": "complementary_info.policy_adapter_id",
    "session_id": "complementary_info.policy_session_id",
    "policy_epoch": "complementary_info.policy_epoch",
    "request_id": "complementary_info.policy_request_id",
    "action_source": "complementary_info.action_source",
}


class RecordingAttributionError(RuntimeError):
    """An episode cannot safely attribute a requested switch or command."""


def resident_recording_features():
    """Explicit feature additions for a newly created recording dataset."""
    return {
        name: {"dtype": "int64" if key == "policy_epoch" else "string",
               "shape": (1,), "names": [key]}
        for key, name in FIELDS.items()
    }


def _context(status):
    task = status.get("task_id")
    if (not isinstance(task, str) or not task
            or task not in status.get("ready", ())
            or task not in status.get("task_metadata", {})):
        raise RecordingAttributionError("Recording task is not an active registered ready adapter")
    session = status.get("session_id")
    epoch = status.get("policy_epoch")
    if (not isinstance(session, str) or not session or isinstance(epoch, bool)
            or not isinstance(epoch, int) or epoch <= 0):
        raise RecordingAttributionError("Recording task has no valid session/epoch identity")
    return {"task_id": task, "adapter_id": task, "session_id": session, "policy_epoch": epoch}


class RecordingEpisodeAttribution:
    def __init__(self):
        self.active = False
        self.failed = False
        self.task_id = None
        self.features = {}
        self.complete = False

    def ensure_usable(self):
        if self.failed:
            raise RecordingAttributionError(
                "Recording episode failed closed; reset cannot resume it. End the recording and create a new client."
            )

    def begin(self, features, status):
        self.ensure_usable()
        if self.active:
            raise RecordingAttributionError("A recording episode is already active")
        present = set(FIELDS.values()).intersection(features)
        if present and present != set(FIELDS.values()):
            raise RecordingAttributionError("Incomplete resident recording identity feature schema")
        canonical_features = {}
        if present:
            # Use the dataset's actual dtype/shape validator; names are descriptive.
            import numpy as np

            from lerobot.datasets.utils import validate_feature_dtype_and_shape

            for key, name in FIELDS.items():
                feature = features[name]
                expected_dtype = "int64" if key == "policy_epoch" else "string"
                shape = feature.get("shape")
                if (feature.get("dtype") != expected_dtype or not isinstance(shape, (tuple, list))
                        or len(shape) != 1 or type(shape[0]) is not int or shape[0] != 1):
                    raise RecordingAttributionError(f"Invalid resident recording feature: {name}")
                # load_info performs this same JSON-list normalization. Keep
                # the caller's dataset immutable; validate our canonical copy.
                canonical_features[name] = deepcopy(feature)
                canonical_features[name]["shape"] = (1,)
                value = np.array([1], dtype=np.int64) if key == "policy_epoch" else "identity"
                error = validate_feature_dtype_and_shape(name, canonical_features[name], value)
                if error:
                    raise RecordingAttributionError(error)
        self.task_id = _context(status)["task_id"]
        self.features = canonical_features
        self.complete = bool(present)
        self.active = True
        return "full_identity" if self.complete else "single_task_only"

    def check_switch(self, task_id):
        self.ensure_usable()
        if self.active and not self.complete and task_id != self.task_id:
            raise RecordingAttributionError(
                "Cannot activate another task inside this episode: dataset lacks all six resident identity columns. "
                "Finish this episode before switching, or create a dataset with resident_recording_features()."
            )

    def frame(self, status, accepted_identity, *, selected_from_policy):
        self.ensure_usable()
        if not self.active:
            raise RecordingAttributionError("Resident action has no active recording episode")
        context = _context(status)
        self.check_switch(context["task_id"])
        if not self.complete:
            return {}
        identity = context | {"request_id": "", "action_source": "human"}
        if selected_from_policy:
            if (not isinstance(accepted_identity, dict)
                    or any(accepted_identity.get(key) != context[key]
                           for key in ("task_id", "session_id", "policy_epoch"))
                    or not isinstance(accepted_identity.get("request_id"), str)
                    or not accepted_identity["request_id"]
                    or accepted_identity.get("accepted_step", 0) <= 0):
                raise RecordingAttributionError("Policy frame lacks its actual confirmed command identity")
            identity.update(request_id=accepted_identity["request_id"], action_source="policy")
        import numpy as np

        from lerobot.datasets.utils import validate_feature_dtype_and_shape

        frame = {FIELDS[key]: np.array([value], dtype=np.int64) if key == "policy_epoch" else value
                 for key, value in identity.items()}
        for name, value in frame.items():
            error = validate_feature_dtype_and_shape(name, self.features[name], value)
            if error:
                raise RecordingAttributionError(error)
        return frame

    def end(self, *, failed=False):
        self.failed = self.failed or bool(failed)
        self.active = False
        self.features = {}
        self.complete = False

