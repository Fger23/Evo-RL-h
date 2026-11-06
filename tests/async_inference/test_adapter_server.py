"""Native adapter server protocol checks: CPU fake policy, real loopback gRPC.

These tests do not instantiate a robot, a real PI05 model, or a CUDA context.
"""

import pickle
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch

from lerobot.async_inference import adapter_policy_server as adapter_entry, policy_server as native
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks

JOINTS = [f"{arm}_{joint}.pos" for arm in ("left", "right") for joint in range(6)]
CAMERAS = ("left_fixed", "left_wrist", "right_wrist")


def features():
    return {
        "observation.state": {"dtype": "float32", "shape": (12,), "names": JOINTS},
        **{
            f"observation.images.{name}": {
                "dtype": "image",
                "shape": (4, 6, 3),
                "names": ["height", "width", "channels"],
            }
            for name in CAMERAS
        },
    }


def policy_specs(**overrides):
    args = {
        "policy_type": "pi05",
        "pretrained_name_or_path": "/server/adapter",
        "lerobot_features": features(),
        "actions_per_chunk": 3,
        "device": "cpu",
        "rename_map": {"observation.state": "renamed.state"},
    }
    args.update(overrides)
    return RemotePolicyConfig(**args)


class FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            image_features={
                f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 6))
                for name in CAMERAS
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
            rtc_config=None,
        )
        self.raw = torch.arange(50 * 12, dtype=torch.float32).reshape(1, 50, 12)
        self.seen = []
        self.rtc_init_count = 0

    def to(self, device):
        assert device == "cpu"
        return self

    def init_rtc_processor(self):
        self.rtc_init_count += 1

    def predict_action_chunk(self, observation, **kwargs):
        self.seen.append((observation, kwargs))
        return self.raw.clone()


@pytest.fixture
def native_loopback(monkeypatch):
    server = native.PolicyServer(
        PolicyServerConfig(host="127.0.0.1", port=8100, inference_latency=0, obs_queue_timeout=0.05)
    )
    model = FakePolicy()
    loaded = []
    processors = []

    def load(spec):
        loaded.append(spec)
        return model

    def make_processors(cfg, **kwargs):
        assert cfg is model.config
        processors.append(kwargs)

        def preprocessor(batch):
            renamed = dict(batch)
            renamed["renamed.state"] = renamed.pop("observation.state")
            return renamed

        return preprocessor, lambda action: action + 1000

    monkeypatch.setattr(server, "_load_policy", load)
    monkeypatch.setattr(native, "make_pre_post_processors", make_processors)
    executor = ThreadPoolExecutor(max_workers=2)
    rpc = grpc.server(executor)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, rpc)
    port = rpc.add_insecure_port("127.0.0.1:0")
    assert port > 0
    rpc.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=5)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        yield SimpleNamespace(server=server, model=model, loaded=loaded, processors=processors, stub=stub)
    finally:
        channel.close()
        rpc.stop(0).wait(timeout=5)
        executor.shutdown(wait=True)


def setup_client(rig, spec):
    rig.stub.Ready(services_pb2.Empty(), timeout=5)
    rig.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(spec)), timeout=5)


def test_default_loader_hook_preserves_original_factory_and_path(monkeypatch):
    expected = object()
    calls = []

    class Factory:
        @classmethod
        def from_pretrained(cls, path):
            calls.append(("path", path))
            return expected

    def get_class(kind):
        calls.append(("type", kind))
        return Factory

    monkeypatch.setattr(native, "get_policy_class", get_class)
    server = native.PolicyServer(PolicyServerConfig())
    assert server._load_policy(policy_specs()) is expected
    assert calls == [("type", "pi05"), ("path", "/server/adapter")]


def test_setup_reuses_native_processors_and_rtc_config(native_loopback):
    rig = native_loopback
    rtc = {"test_marker": "preserved"}
    setup_client(rig, policy_specs(rtc_config=rtc))
    assert len(rig.loaded) == 1
    assert rig.loaded[0].pretrained_name_or_path == "/server/adapter"
    assert rig.model.config.rtc_config == rtc
    assert rig.model.rtc_init_count == 1
    assert rig.processors == [
        {
            "pretrained_path": "/server/adapter",
            "preprocessor_overrides": {
                "device_processor": {"device": "cpu"},
                "rename_observations_processor": {"rename_map": {"observation.state": "renamed.state"}},
            },
            "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
        }
    ]


def test_native_wire_returns_twelve_dimensional_raw_and_processed_actions(native_loopback):
    rig = native_loopback
    setup_client(rig, policy_specs())
    raw = {name: float(i) for i, name in enumerate(JOINTS)}
    raw.update({name: np.full((4, 6, 3), 255, dtype=np.uint8) for name in CAMERAS})
    raw.update(
        {
            "task": "straighten cash",
            native.RTC_INFERENCE_DELAY_KEY: 2,
            native.RTC_PREV_CHUNK_LEFT_OVER_KEY: torch.ones(1, 2, 12),
            native.RTC_EXECUTION_HORIZON_KEY: 3,
        }
    )
    timestamp = 1000.0
    observation = TimedObservation(timestamp=timestamp, timestep=7, observation=raw, must_go=True)
    rig.stub.SendObservations(
        send_bytes_in_chunks(pickle.dumps(observation), services_pb2.Observation), timeout=5
    )
    response = rig.stub.GetActions(services_pb2.Empty(), timeout=5)
    assert response.data, "Native inference returned no action bytes"
    actions = pickle.loads(response.data)
    assert len(actions) == 3
    for i, action in enumerate(actions):
        assert action.get_timestep() == 7 + i
        assert action.get_timestamp() == pytest.approx(timestamp + i / 30)
        torch.testing.assert_close(action.get_original_action(), rig.model.raw[0, i])
        torch.testing.assert_close(action.get_action(), rig.model.raw[0, i] + 1000)
        assert action.get_action().shape == (12,)
    batch, kwargs = rig.model.seen[0]
    torch.testing.assert_close(batch["renamed.state"], torch.arange(12, dtype=torch.float32).reshape(1, 12))
    assert batch["task"] == "straighten cash"
    for name in CAMERAS:
        image = batch[f"observation.images.{name}"]
        assert image.shape == (1, 3, 4, 6)
        torch.testing.assert_close(image, torch.ones_like(image))
    assert kwargs["inference_delay"] == 2
    assert kwargs["execution_horizon"] == 3
    torch.testing.assert_close(kwargs["prev_chunk_left_over"], torch.ones(1, 2, 12))
    assert not any(key.startswith("__rtc_") for key in batch)


def test_model_loading_failure_is_a_failed_setup_not_a_ready_policy(native_loopback, monkeypatch):
    rig = native_loopback

    def fail(spec):
        raise ValueError("adapter checkpoint mismatch")

    monkeypatch.setattr(rig.server, "_load_policy", fail)
    rig.stub.Ready(services_pb2.Empty(), timeout=5)
    with pytest.raises(grpc.RpcError, match="adapter checkpoint mismatch"):
        rig.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_specs())), timeout=5
        )
    assert rig.server.policy is None
    assert rig.processors == []


def test_adapter_entry_inherits_original_rpc_and_chunk_methods():
    for name in (
        "Ready",
        "SendPolicyInstructions",
        "SendObservations",
        "GetActions",
        "_predict_action_chunk",
        "_get_action_chunk",
        "_enqueue_observation",
        "_time_action_chunk",
    ):
        assert getattr(adapter_entry.AdapterPolicyServer, name) is getattr(native.PolicyServer, name)


def test_adapter_loader_only_changes_model_construction(tmp_path, monkeypatch):
    adapter_path = str(tmp_path / "A")
    cfg = adapter_entry.AdapterPolicyServerConfig(adapter_path=adapter_path, base_path="/same/base")
    server = adapter_entry.AdapterPolicyServer(cfg)
    expected = object()
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(adapter_entry, "load_adapter_policy", loader)
    spec = policy_specs(pretrained_name_or_path=adapter_path, device="cuda:0")
    assert server._load_policy(spec) is expected
    assert calls == [(adapter_path, {"device": "cuda:0", "base_override": "/same/base"})]


def test_adapter_processor_override_adds_only_saved_local_tokenizer(tmp_path):
    adapter_path = tmp_path / "A"
    spec = policy_specs(pretrained_name_or_path=str(adapter_path), device="cuda:0")
    server = adapter_entry.AdapterPolicyServer(
        adapter_entry.AdapterPolicyServerConfig(adapter_path=str(adapter_path))
    )
    server.device = "cuda:0"
    default = native.PolicyServer._preprocessor_overrides(server, spec)
    assert default == {
        "device_processor": {"device": "cuda:0"},
        "rename_observations_processor": {"rename_map": spec.rename_map},
    }
    overridden = server._preprocessor_overrides(spec)
    assert overridden == {
        **default,
        "tokenizer_processor": {"tokenizer_name": str(adapter_path.resolve() / "tokenizer")},
    }
    assert "tokenizer_processor" not in default


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"policy_type": "act"}, "policy_type=pi05"),
        ({"pretrained_name_or_path": "/another/adapter"}, "must match"),
        ({"device": "cuda:1"}, "cuda:0"),
        ({"device": "cpu"}, "cuda:0"),
    ],
)
def test_adapter_request_rejected_before_model_load(tmp_path, monkeypatch, overrides, message):
    adapter_path = str(tmp_path / "A")
    server = adapter_entry.AdapterPolicyServer(
        adapter_entry.AdapterPolicyServerConfig(adapter_path=adapter_path)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid request reached the model loader")

    monkeypatch.setattr(adapter_entry, "load_adapter_policy", forbidden)
    spec = {"pretrained_name_or_path": adapter_path, "device": "cuda:0"}
    spec.update(overrides)
    with pytest.raises(ValueError, match=message):
        server._load_policy(policy_specs(**spec))


def test_gpu_selection_binds_authorized_uuid_before_cuda_context(monkeypatch):
    calls = []
    monkeypatch.setattr(adapter_entry.torch.cuda, "is_initialized", lambda: False)

    def nvidia_smi(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=adapter_entry.AUTHORIZED_GPU_UUID + "\n")

    monkeypatch.setattr(adapter_entry.subprocess, "run", nvidia_smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not_the_selected_device")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "not_the_selected_order")
    adapter_entry.select_gpu(adapter_entry.AdapterPolicyServerConfig())
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == adapter_entry.AUTHORIZED_GPU_UUID
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert calls == [["nvidia-smi", "--id=1", "--query-gpu=uuid", "--format=csv,noheader"]]


@pytest.mark.parametrize("kind", ["wrong_index", "wrong_uuid", "initialized", "different_hardware"])
def test_gpu_selection_refuses_unauthorized_or_already_initialized_device(monkeypatch, kind):
    cfg = adapter_entry.AdapterPolicyServerConfig()
    if kind == "wrong_index":
        cfg.gpu = 0
    elif kind == "wrong_uuid":
        cfg.gpu_uuid = "GPU-unapproved"
    monkeypatch.setattr(adapter_entry.torch.cuda, "is_initialized", lambda: kind == "initialized")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preserve_me")

    def nvidia_smi(argv, **kwargs):
        assert kind == "different_hardware", "Bad config must be rejected before querying hardware"
        return SimpleNamespace(stdout="GPU-not-the-authorized-card\n")

    monkeypatch.setattr(adapter_entry.subprocess, "run", nvidia_smi)
    with pytest.raises((ValueError, RuntimeError)):
        adapter_entry.select_gpu(cfg)
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "preserve_me"
