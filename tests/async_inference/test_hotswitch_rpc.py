"""Real localhost gRPC tests with deterministic CPU policy and no robot."""

import pickle
import threading
import time
import unittest
from concurrent import futures

import grpc

from lerobot.async_inference.hotswitch import (
    PROTOCOL, HotSwitchRuntime, InferenceRequest, PolicyIdentity,
)
from lerobot.transport import services_pb2, services_pb2_grpc


class RPCBackend:
    task_ids = ("A", "B")

    def __init__(self):
        self.active = None
        self.entered = None
        self.release = None

    def metadata(self, task):
        return {"task_text": task, "rtc_mode": "async"}

    def ready(self, task):
        return True

    def activate(self, task):
        self.active = task

    def reset(self, task):
        pass

    def predict(self, task, observation, prefix_steps=0, action_prefix=None):
        if self.entered:
            self.entered.set()
            if not self.release.wait(timeout=3):
                raise RuntimeError("test synchronization timeout")
        if task != self.active:
            raise RuntimeError("task changed while model running")
        return [[1.0]], [[10.0]]


class HotSwitchRPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.policy_server import PolicyServer

        cls.PolicyServerConfig, cls.PolicyServer = PolicyServerConfig, PolicyServer

    def setUp(self):
        self.backend = RPCBackend()
        self.runtime = HotSwitchRuntime(self.backend)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        self.servicer = self.PolicyServer(self.PolicyServerConfig(), hotswitch_runtime=self.runtime)
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(self.servicer, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        grpc.channel_ready_future(self.channel).result(timeout=5)
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.state = self.control(op="connect", operation_id="connect", task_id="A")["state"]

    def tearDown(self):
        if self.backend.release:
            self.backend.release.set()
        self.channel.close()
        self.server.stop(0).wait(timeout=5)

    def control(self, timeout=3, **values):
        envelope = {"protocol": PROTOCOL, "version": 1, **values}
        response = self.stub.ControlPolicy(services_pb2.PolicySetup(data=pickle.dumps(envelope)), timeout=timeout)
        return pickle.loads(response.data)

    def request(self, request_id="req"):
        return InferenceRequest(
            PolicyIdentity(self.state["session_id"], self.state["task_id"], self.state["policy_epoch"], request_id),
            {"offline": True},
        )

    def test_capability_identity_prediction_and_stale_rejection(self):
        capability = self.control(op="capabilities")
        self.assertIn("hotswitch_v1", capability["capabilities"])
        request = self.request()
        response = pickle.loads(self.stub.Predict(services_pb2.PolicySetup(data=pickle.dumps(request)), timeout=3).data)
        self.assertEqual(response.identity, request.identity)
        self.assertEqual(response.processed_actions, [[10.0]])
        reply = self.control(op="switch", operation_id="B", task_id="B", session_id=self.state["session_id"])
        self.assertTrue(reply["ok"])
        late = pickle.loads(self.stub.Predict(services_pb2.PolicySetup(data=pickle.dumps(request)), timeout=3).data)
        self.assertIn("Stale", late.error)
        self.assertEqual(late.identity, request.identity)

    def test_timeout_retry_and_status_resolve_single_commit(self):
        self.backend.entered, self.backend.release = threading.Event(), threading.Event()
        inference = self.stub.Predict.future(services_pb2.PolicySetup(data=pickle.dumps(self.request())), timeout=5)
        self.assertTrue(self.backend.entered.wait(timeout=2))
        with self.assertRaises(grpc.RpcError) as caught:
            self.control(timeout=0.03, op="switch", operation_id="timeout-switch", task_id="B", session_id=self.state["session_id"])
        self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
        self.backend.release.set()
        inferred = pickle.loads(inference.result(timeout=3).data)
        self.assertIsNone(inferred.error)
        retry = self.control(op="switch", operation_id="timeout-switch", task_id="B", session_id=self.state["session_id"])
        self.assertTrue(retry["ok"])
        state = self.control(op="status")["state"]
        self.assertEqual(state["task_id"], "B")
        self.assertEqual(state["policy_epoch"], self.state["policy_epoch"] + 1)

    def test_legacy_calls_rejected_on_registry_server(self):
        with self.assertRaises(grpc.RpcError) as caught:
            self.stub.Ready(services_pb2.Empty(), timeout=2)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_runtime_path_strings_and_bad_versions_are_rejected(self):
        reply = self.control(op="switch", operation_id="bad", task_id="E:/untrusted/model", session_id=self.state["session_id"])
        self.assertFalse(reply["ok"])
        self.assertIn("not registered", reply["error"])
        reply = self.control(op="status", version=99)
        self.assertFalse(reply["ok"])


if __name__ == "__main__":
    unittest.main()
