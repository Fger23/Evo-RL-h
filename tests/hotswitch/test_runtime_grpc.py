"""Loopback test of the actual generated stubs and PolicyServer extension.

Uses simulated policies and an in-memory simulated control interface only.
"""

import unittest
import uuid
from concurrent import futures

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.hotswitch.runtime import (
    GrpcHotSwitchTransport,
    HotSwitchClientCore,
    HotSwitchRuntime,
)
from lerobot.async_inference.hotswitch.simulation import SimulatedBackend, SimulatedRobot
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc


class GrpcTests(unittest.TestCase):
    def setUp(self):
        self.runtime = HotSwitchRuntime(SimulatedBackend(delay_s=0))
        self.runtime.preload()
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ],
        )
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(
            PolicyServer(PolicyServerConfig(), hotswitch_runtime=self.runtime), self.server
        )
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.transport = GrpcHotSwitchTransport(f"127.0.0.1:{port}", timeout=3)
        self.robot = SimulatedRobot()
        self.client = HotSwitchClientCore(self.transport, self.robot.hold)
        self.client.connect("A")

    def tearDown(self):
        self.transport.close()
        self.server.stop(0).wait(3)

    def test_actual_loopback_control_boundary_switch_and_raw_refill(self):
        self.client.request_actions({})
        first = self.client.next_action()
        self.client.confirm_action(self.robot.send_action(first))
        self.client.request_actions({})
        self.assertEqual(len(self.client.queue), 8)
        self.transport.request_switch("B", uuid.uuid4().hex, self.client.status["session_id"])
        self.assertEqual(self.transport.status()["task_id"], "A")
        self.client.control_boundary()
        self.client.request_actions({})
        self.assertNotEqual(self.client.next_action(), first)
        self.assertEqual(self.client.status["task_id"], "B")

    def test_old_setup_is_rejected_on_resident_server(self):
        with self.assertRaises(grpc.RpcError) as error:
            self.transport.stub.Ready(services_pb2.Empty(), timeout=3)
        self.assertEqual(error.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_version_negotiation_rejects_unknown_version(self):
        with self.assertRaises(grpc.RpcError) as error:
            self.transport.stub.HotSwitchSession(
                services_pb2.HotSwitchMessage(protocol_version=77), timeout=3
            )
        self.assertEqual(error.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    def test_only_registered_ids_allowed(self):
        with self.assertRaises(grpc.RpcError):
            self.transport.request_switch("/tmp/arbitrary-model", "bad", self.client.status["session_id"])
        self.assertEqual(self.transport.status()["task_id"], "A")

    def test_observation_and_response_larger_than_default_grpc_limit(self):
        # Three unresized float32 cameras can exceed gRPC's default 4 MiB.
        payload = "x" * (5 * 1024 * 1024)
        self.runtime.backend.task_metadata["A"]["test_large_metadata"] = payload
        self.assertEqual(
            len(self.transport.status()["task_metadata"]["A"]["test_large_metadata"]), len(payload)
        )
        self.assertTrue(self.client.request_actions({"large_observation": payload}))


if __name__ == "__main__":
    unittest.main()
