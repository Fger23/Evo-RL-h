"""Optional single-device proof that joint LoRA never changes Base parameters."""

import hashlib
import json
import uuid
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_hashes(snapshot):
    return {name: group["sha256"] for name, group in snapshot["adapter_groups"].items()}


def checkpoint_artifact_manifest(checkpoint_dir):
    """Hash saved model/tokenizer/processors and training state, excluding this proof."""
    root = Path(checkpoint_dir).resolve(strict=True)
    artifacts = {}
    for directory in ("pretrained_model", "training_state"):
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Audited checkpoint is missing {directory}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in ("training_state/peft_base_audit.json", "training_state/peft_base_audit.tmp"):
                continue
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"Checkpoint artifact escapes saved directory: {relative}")
            artifacts[relative] = _file_sha256(path)
    if not any(name.startswith("pretrained_model/") for name in artifacts):
        raise ValueError("Audited checkpoint has no saved policy artifacts")
    step = json.loads((root / "training_state/training_step.json").read_text(encoding="utf-8"))["step"]
    return {"path": str(root), "training_step": step, "artifacts": artifacts}


def validate_audit_coverage(
    report_or_path,
    expected_end=None,
    *,
    required_groups=None,
    require_final=False,
    require_checkpoint=True,
    expected_checkpoint=None,
    _seen=None,
):
    """Verify continuous 0..end coverage and every referenced checkpoint proof.

    This validator imports no model framework. A resumed attempt's absolute end
    step is never treated as the number of updates performed by that attempt.
    """
    seen = set() if _seen is None else set(_seen)
    source_path = None
    if isinstance(report_or_path, (str, Path)):
        path = Path(report_or_path).resolve(strict=True)
        source_path = path
        if path in seen:
            raise ValueError("Cyclic frozen Base audit history")
        seen.add(path)
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = report_or_path
    allowed = ("PASSED",) if require_final else ("PASSED", "CHECKPOINT_PASSED")
    if (
        report.get("schema_version") != 2
        or report.get("status") not in allowed
        or report.get("coverage_complete") is not True
    ):
        raise ValueError("Frozen Base audit has no completed continuous coverage")
    end = report.get("steps_completed")
    start = report.get("start_step")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError("Frozen Base audit has an invalid or empty training interval")
    if expected_end is not None and end != expected_end:
        raise ValueError("Frozen Base audit coverage ends at a different training step")
    if report.get("updates_this_attempt") != end - start:
        raise ValueError("Absolute training step cannot replace this attempt's update count")
    groups = list(report.get("required_groups", []))
    if len(groups) != len(set(groups)) or (
        required_groups is not None and set(groups) != set(required_groups)
    ):
        raise ValueError("Frozen Base audit required branches differ")
    before, after = report.get("before", {}), report.get("after", {})
    if not before.get("parameter_inventory") or any(
        not snapshot.get("adapter_groups")
        or any(group.get("finite") is not True for group in snapshot["adapter_groups"].values())
        for snapshot in (before, after)
    ):
        raise ValueError("Frozen Base audit inventory/finite adapter evidence is missing")
    base_sha = before.get("sha256", "")
    if (
        len(base_sha) != 64
        or any(c not in "0123456789abcdef" for c in base_sha)
        or base_sha != after.get("sha256")
        or report.get("frozen_base_unchanged") is not True
        or _canonical(before.get("parameter_inventory")) != _canonical(after.get("parameter_inventory"))
    ):
        raise ValueError("Frozen Base content/inventory changed or is absent")
    coverage = report.get("coverage")
    if not isinstance(coverage, list) or not coverage:
        raise ValueError("Frozen Base audit coverage is missing")
    cursor = 0
    previous_adapter_hashes = None
    for segment in coverage:
        if (
            segment.get("start_step") != cursor
            or not isinstance(segment.get("end_step"), int)
            or segment["end_step"] <= cursor
            or segment.get("updates") != segment["end_step"] - cursor
            or segment.get("base_sha256") != base_sha
            or segment.get("frozen_base_unchanged") is not True
            or set(segment.get("required_groups", [])) != set(groups)
            or segment.get("adapter_values_finite") is not True
            or any(segment.get("adapter_groups_updated", {}).get(group) is not True for group in groups)
        ):
            raise ValueError("Frozen Base audit coverage has a gap, overlap, or invalid update segment")
        initial, final = (
            segment.get("before_adapter_groups_sha256"),
            segment.get("after_adapter_groups_sha256"),
        )
        if (
            not initial
            or not final
            or (previous_adapter_hashes is not None and initial != previous_adapter_hashes)
        ):
            raise ValueError("Adapter state is not continuous across audited resume segments")
        if any(
            group not in initial or group not in final or initial[group] == final[group] for group in groups
        ):
            raise ValueError("Required LoRA branch did not update within an audited segment")
        previous_adapter_hashes = final
        cursor = segment["end_step"]
    current = coverage[-1]
    if (
        cursor != end
        or current["start_step"] != start
        or current.get("audit_id") != report.get("audit_id")
        or current["before_adapter_groups_sha256"] != _group_hashes(before)
        or current["after_adapter_groups_sha256"] != _group_hashes(after)
        or report.get("adapter_values_finite") is not True
        or any(report.get("adapter_groups_updated", {}).get(group) is not True for group in groups)
    ):
        raise ValueError("Current attempt does not match its coverage proof")
    previous = report.get("previous_report")
    if start:
        if not previous or not previous.get("path") or not previous.get("sha256"):
            raise ValueError("Resumed audit requires its actual checkpoint report reference")
        previous_path = Path(previous["path"]).resolve(strict=True)
        if _file_sha256(previous_path) != previous["sha256"]:
            raise ValueError("Previous checkpoint audit content hash changed")
        validate_audit_coverage(
            previous_path, start, required_groups=groups, require_checkpoint=True, _seen=seen
        )
        recorded = json.loads(previous_path.read_text(encoding="utf-8"))
        if (
            _canonical(recorded["coverage"]) != _canonical(coverage[:-1])
            or recorded["after"]["sha256"] != base_sha
            or _canonical(recorded["after"]["parameter_inventory"])
            != _canonical(before["parameter_inventory"])
            or _group_hashes(recorded["after"]) != _group_hashes(before)
        ):
            raise ValueError("Previous checkpoint proof differs from the resumed model state")
    elif previous is not None or len(coverage) != 1:
        raise ValueError("An initial audit cannot claim an unrelated previous segment")
    if require_checkpoint:
        binding = report.get("checkpoint")
        if not binding or not binding.get("path"):
            raise ValueError("Audit is not bound to actual saved checkpoint artifacts")
        if (
            source_path is not None
            and report["status"] == "CHECKPOINT_PASSED"
            and source_path
            != (Path(binding["path"]) / "training_state/peft_base_audit.json").resolve(strict=True)
        ):
            raise ValueError("Checkpoint sidecar does not belong to its declared saved directory")
        if expected_checkpoint is not None and Path(binding["path"]).resolve(strict=True) != Path(
            expected_checkpoint
        ).resolve(strict=True):
            raise ValueError("Audit is bound to a different checkpoint than the one selected")
        actual = checkpoint_artifact_manifest(binding["path"])
        if actual["training_step"] != end or _canonical(actual) != _canonical(binding):
            raise ValueError("Saved checkpoint artifact bytes changed, disappeared or differ from audit")
        if report["status"] == "PASSED":
            final = report.get("final_checkpoint_report")
            if not final or not final.get("path") or not final.get("sha256"):
                raise ValueError("Final audit lacks its actual checkpoint sidecar reference")
            final_path = Path(final["path"]).resolve(strict=True)
            if final_path != (Path(binding["path"]) / "training_state/peft_base_audit.json").resolve(
                strict=True
            ):
                raise ValueError("Final report does not reference its actual checkpoint sidecar")
            if _file_sha256(final_path) != final["sha256"]:
                raise ValueError("Final checkpoint audit sidecar content hash changed")
            validate_audit_coverage(
                final_path, end, required_groups=groups, require_checkpoint=True, _seen=seen
            )
            sidecar = json.loads(final_path.read_text(encoding="utf-8"))
            if sidecar.get("status") != "CHECKPOINT_PASSED" or any(
                _canonical(sidecar.get(key)) != _canonical(report.get(key))
                for key in ("audit_id", "start_step", "before", "after", "coverage", "checkpoint")
            ):
                raise ValueError("Final report differs from the actual saved checkpoint proof")
    return {
        "start_step": 0,
        "end_step": end,
        "segments": len(coverage),
        "base_sha256": base_sha,
        "updates_this_attempt": end - start,
        "coverage_complete": True,
    }


def frozen_parameter_snapshot(policy):
    import torch

    if not getattr(policy, "peft_config", None) or not callable(getattr(policy, "get_base_model", None)):
        raise ValueError("Frozen Base audit requires a restored or newly wrapped PEFT model")
    base = policy.get_base_model()
    digest = hashlib.sha256()
    inventory = []
    adapters = []
    adapter_groups = {}
    for name, parameter in base.named_parameters():
        if "lora_" in name:
            if parameter.requires_grad:
                adapters.append(name)
                if ".paligemma." in name and ".language_model." in name:
                    group = "vlm"
                elif ".gemma_expert." in name:
                    group = "expert"
                else:
                    group = next(
                        (
                            key
                            for key in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
                            if f".{key}." in f".{name}"
                        ),
                        "other",
                    )
                entry = adapter_groups.setdefault(
                    group, {"digest": hashlib.sha256(), "names": [], "finite": True}
                )
                entry["digest"].update(name.encode())
                entry["digest"].update(
                    parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                )
                entry["names"].append(name)
                entry["finite"] = entry["finite"] and bool(torch.isfinite(parameter.detach()).all())
            continue
        if parameter.requires_grad:
            raise ValueError(f"Unexpected trainable Base parameter: {name}")
        inventory.append((name, tuple(parameter.shape), str(parameter.dtype)))
        digest.update(name.encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(json.dumps(list(parameter.shape)).encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not adapters or not inventory:
        raise ValueError("Audit requires both trainable LoRA tensors and frozen Base tensors")
    return {
        "sha256": digest.hexdigest(),
        "parameter_inventory": inventory,
        "trainable_adapter_parameters": adapters,
        "adapter_groups": {
            key: {"sha256": entry["digest"].hexdigest(), "names": entry["names"], "finite": entry["finite"]}
            for key, entry in adapter_groups.items()
        },
        "scope": "Frozen parameters only; LoRA tensors and runtime buffers excluded",
    }


class FrozenPeftBaseAudit:
    """Track a real training interval and optional saved-checkpoint evidence.

    The trainer sets require_checkpoint_binding=True. Legacy standalone tensor
    checks may finish as NOT_CHECKPOINTED, which the formal validator rejects.
    """

    def __init__(
        self,
        policy,
        output_dir,
        required_groups=(),
        *,
        start_step=0,
        previous_report=None,
        require_checkpoint_binding=False,
    ):
        self.before = frozen_parameter_snapshot(policy)
        if not all(group["finite"] for group in self.before["adapter_groups"].values()):
            raise ValueError("Initial LoRA parameters contain NaN/Inf")
        self.required_groups = tuple(required_groups)
        if any(key not in self.before["adapter_groups"] for key in self.required_groups):
            raise ValueError("Required joint LoRA branches are missing from the audit")
        if not isinstance(start_step, int) or start_step < 0:
            raise ValueError("Audit start_step must be the actual nonnegative loaded training step")
        self.start_step = start_step
        self.require_checkpoint_binding = require_checkpoint_binding
        self.last_checkpoint = None
        self.previous = None
        self.previous_coverage = []
        if start_step:
            if previous_report is None:
                raise ValueError("Resume requires the corresponding checkpoint frozen Base audit")
            previous_path = Path(previous_report).resolve(strict=True)
            recorded_step = json.loads(
                (previous_path.parent / "training_step.json").read_text(encoding="utf-8")
            )["step"]
            if recorded_step != start_step:
                raise ValueError("Actual checkpoint training step differs from audit resume start")
            validate_audit_coverage(
                previous_path,
                start_step,
                required_groups=self.required_groups,
                expected_checkpoint=previous_path.parent.parent,
            )
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if (
                previous["after"]["sha256"] != self.before["sha256"]
                or _canonical(previous["after"]["parameter_inventory"])
                != _canonical(self.before["parameter_inventory"])
                or _group_hashes(previous["after"]) != _group_hashes(self.before)
            ):
                raise ValueError("Loaded Base/adapter differs from the audited resume checkpoint")
            self.previous = {"path": str(previous_path), "sha256": _file_sha256(previous_path)}
            self.previous_coverage = previous["coverage"]
        elif previous_report is not None:
            raise ValueError("Initial training at step zero must not claim a previous audit")
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.audit_id = uuid.uuid4().hex
        self.report = {
            "schema_version": 2,
            "status": "STARTED",
            "audit_id": self.audit_id,
            "before": self.before,
            "required_groups": self.required_groups,
            "start_step": start_step,
            "steps_completed": start_step,
            "updates_this_attempt": 0,
            "previous_report": self.previous,
            "coverage": self.previous_coverage,
            "coverage_complete": False,
            "checkpoint_binding_status": "NOT_CHECKPOINTED",
        }
        self._save()

    def _save(self):
        # Each attempt retains its own progress report. Completed checkpoint
        # proofs are separate immutable files, and the latest pointer is atomic.
        for name in (f"peft_base_audit_{self.audit_id}.json", "peft_base_audit.json"):
            path = self.output / name
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            temporary.replace(path)

    def _assess(self, policy, steps, status):
        try:
            if not isinstance(steps, int) or steps <= self.start_step:
                raise AssertionError("No new training updates in this audit attempt")
            after = frozen_parameter_snapshot(policy)
            self.report.update(
                after=after, steps_completed=steps, updates_this_attempt=steps - self.start_step
            )
            unchanged = after["sha256"] == self.before["sha256"] and _canonical(
                after["parameter_inventory"]
            ) == _canonical(self.before["parameter_inventory"])
            updates = {
                key: (
                    key in after["adapter_groups"]
                    and after["adapter_groups"][key]["sha256"] != self.before["adapter_groups"][key]["sha256"]
                )
                for key in self.required_groups
            }
            finite = all(group["finite"] for group in after["adapter_groups"].values())
            passed = unchanged and finite and all(updates.values())
            self.report.update(
                status=status if passed else "FAILED",
                frozen_base_unchanged=unchanged,
                adapter_groups_updated=updates,
                adapter_values_finite=finite,
            )
            if not unchanged:
                raise AssertionError("Frozen Base parameters changed during PEFT training")
            if not finite or not all(updates.values()):
                raise AssertionError("Joint LoRA branches did not all update with finite values")
            segment = {
                "audit_id": self.audit_id,
                "start_step": self.start_step,
                "end_step": steps,
                "updates": steps - self.start_step,
                "base_sha256": self.before["sha256"],
                "frozen_base_unchanged": unchanged,
                "required_groups": list(self.required_groups),
                "adapter_groups_updated": updates,
                "adapter_values_finite": finite,
                "before_adapter_groups_sha256": _group_hashes(self.before),
                "after_adapter_groups_sha256": _group_hashes(after),
            }
            self.report.update(coverage=[*self.previous_coverage, segment], coverage_complete=True)
            validate_audit_coverage(
                self.report, steps, required_groups=self.required_groups, require_checkpoint=False
            )
        except Exception as error:
            self.report.setdefault("frozen_base_unchanged", False)
            self.report.update(status="FAILED", error=str(error), coverage_complete=False)
            self._save()
            raise
        return self.report

    def checkpoint(self, policy, steps, checkpoint_dir):
        """Called after save_checkpoint, before publishing the last-checkpoint link."""
        try:
            path = (Path(checkpoint_dir) / "training_state/peft_base_audit.json").resolve()
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint audit: {path}")
            self._assess(policy, steps, "CHECKPOINT_PASSED")
            self.report.pop("final_checkpoint_report", None)
            self.report.update(
                checkpoint=checkpoint_artifact_manifest(checkpoint_dir),
                checkpoint_binding_status="CHECKPOINT_VERIFIED",
            )
            validate_audit_coverage(self.report, steps, required_groups=self.required_groups)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            temporary.replace(path)
            self.last_checkpoint = {
                "path": str(path),
                "sha256": _file_sha256(path),
                "training_step": steps,
                "checkpoint": self.report["checkpoint"],
            }
            self._save()
        except Exception as error:
            self.report.update(status="FAILED", error=str(error), coverage_complete=False)
            self._save()
            raise
        return self.report

    def finish(self, policy, steps):
        try:
            self._assess(policy, steps, "PASSED")
            if self.last_checkpoint is not None and self.last_checkpoint["training_step"] == steps:
                self.report.update(
                    checkpoint=self.last_checkpoint["checkpoint"],
                    final_checkpoint_report={key: self.last_checkpoint[key] for key in ("path", "sha256")},
                    checkpoint_binding_status="FINAL_CHECKPOINT_BOUND",
                )
                validate_audit_coverage(
                    self.report, steps, required_groups=self.required_groups, require_final=True
                )
            else:
                self.report.pop("checkpoint", None)
                self.report.pop("final_checkpoint_report", None)
                self.report["checkpoint_binding_status"] = "NOT_CHECKPOINTED"
                if self.require_checkpoint_binding:
                    raise ValueError("Final audit requires an actually saved and verified final checkpoint")
            self._save()
        except Exception as error:
            self.report.update(status="FAILED", error=str(error), coverage_complete=False)
            self._save()
            raise
        return self.report
