"""Validate this guide's wiring with CPU-only gRPC and an inert owned child.

No model is constructed, no GPU is allocated, and no robot driver is imported.
Run only with the activated personal environment; evidence lives on personal NFS.
"""
import argparse
from concurrent import futures
import importlib.util
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[2]
TOOLS = CODE / "scripts/hotswitch"
sys.path[:0] = [str(TOOLS), str(CODE / "examples/hotswitch")]
EVIDENCE = None


class GuideTests(unittest.TestCase):
    def setUp(self):
        import grpc
        from lerobot.async_inference.configs import PolicyServerConfig
        from lerobot.async_inference.hotswitch.runtime import HotSwitchRuntime
        from lerobot.async_inference.hotswitch.simulation import SimulatedBackend
        from lerobot.async_inference.policy_server import PolicyServer
        from lerobot.transport import services_pb2_grpc

        self.tmp = tempfile.TemporaryDirectory(dir=EVIDENCE)
        self.directory = Path(self.tmp.name)
        backend = SimulatedBackend()
        backend.task_ids = ('A', 'A_copy')
        backend.task_metadata['A_copy'] = backend.task_metadata.pop('B')
        self.runtime = HotSwitchRuntime(backend)
        self.runtime.preload()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        port = self.server.add_insecure_port('127.0.0.1:0')
        self.address = f'127.0.0.1:{port}'
        servicer = PolicyServer(PolicyServerConfig(host='127.0.0.1', port=port), hotswitch_runtime=self.runtime)
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, self.server)
        self.server.start()

    def tearDown(self):
        self.server.stop(0).wait(timeout=10)
        self.tmp.cleanup()

    def test_infer_returns_predictions_without_sending_actions(self):
        from infer_client import infer_once
        with patch('lerobot.async_inference.hotswitch.simulation.SimulatedRobot.send_action',
                   side_effect=AssertionError('Inference-only client must not send commands')):
            value = infer_once(self.address, 'A', {'offline': True})
        self.assertEqual(value['status'], 'INFERENCE_CHUNK_RECEIVED')
        self.assertEqual((value['chunk_length'], value['action_dim']), (8, 2))
        self.assertEqual(value['prefix_steps'], 0)
        self.assertEqual(value['robot_commands_sent'], 0)
        self.assertEqual(self.runtime.metrics['inferences'], 1)
        self.assertEqual(value['raw_actions'][0], [.2, .2])
        self.assertEqual(value['processed_actions'][0], [.4, .4])

    def test_existing_session_is_not_automatically_taken_over(self):
        from infer_client import infer_once
        first = infer_once(self.address, 'A', {'offline': True})
        with self.assertRaisesRegex(RuntimeError, 'existing session'):
            infer_once(self.address, 'A', {'offline': True})
        self.assertEqual(self.runtime.session_id, first['session_id'])
        self.assertEqual(self.runtime.metrics['inferences'], 1)

    def test_explicit_reconnect_creates_new_session(self):
        from infer_client import infer_once
        first = infer_once(self.address, 'A', {'offline': True})
        second = infer_once(self.address, 'A_copy', {'offline': True}, previous_session_id=first['session_id'])
        self.assertNotEqual(first['session_id'], second['session_id'])
        self.assertEqual(second['task_id'], 'A_copy')
        self.assertEqual(self.runtime.metrics['inferences'], 2)

    def test_wait_ready_checks_real_grpc(self):
        from wait_ready import wait_ready
        (self.directory / 'ready.json').write_text(json.dumps({'address': self.address}))
        self.assertEqual(wait_ready(self.directory, timeout=3), self.address)
        self.assertIsNone(self.runtime.session_id)

    def test_wait_ready_reports_failed_guardian(self):
        from wait_ready import wait_ready
        (self.directory / 'guardian_state.json').write_text(json.dumps({'status': 'FAILED', 'error': 'CPU_FIXTURE'}))
        with self.assertRaisesRegex(RuntimeError, 'CPU_FIXTURE'):
            wait_ready(self.directory, timeout=3)

    def test_client_plan_does_not_contact_server_or_write(self):
        output = self.directory / 'must_not_exist.json'
        args = [sys.executable, '-B', str(CODE / 'examples/hotswitch/infer_client.py'), '--address', self.address,
                '--observation', str(self.directory / 'nonexistent.safetensors'), '--output', str(output)]
        result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=20)
        self.assertEqual(json.loads(result.stdout)['status'], 'PLAN_ONLY_NOT_EXECUTED')
        self.assertFalse(output.exists())
        self.assertIsNone(self.runtime.session_id)

    def test_stop_wrapper_reaps_only_its_inert_child(self):
        root = Path(os.environ['EVORL_INFER_ROOT'])
        root.mkdir(exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix='guide_cpu_', dir=root))
        (session / 'server_work').mkdir()
        token = secrets.token_hex(24)
        spec = {'schema_version': 1,
                'argv': [sys.executable, '-c', 'import time; time.sleep(60)'],
                'cwd': str(session / 'server_work'),
                'stdout': str(session / 'server.stdout.log'), 'stderr': str(session / 'server.stderr.log'),
                'expected_uid': os.getuid(), 'harness_pid': os.getpid(),
                'environment': {'CUDA_VISIBLE_DEVICES': '', 'HOTSWITCH_RPC_RUN_TOKEN': token,
                                'PYTHONPATH': os.environ['PYTHONPATH']}}
        for name, data in [('guardian.token', token), ('server_spec.json', json.dumps(spec))]:
            fd = os.open(session / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                stream.write(data)
        guardian_path = TOOLS / 'server_guardian.py'
        module_spec = importlib.util.spec_from_file_location('guide_test_guardian', guardian_path)
        guardian = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(guardian)
        with (session / 'guardian.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(guardian_path), 'serve',
                                      '--spec', str(session / 'server_spec.json'),
                                      '--state', str(session / 'guardian_state.json'),
                                      '--token-file', str(session / 'guardian.token')],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                path = session / 'guardian_state.json'
                if path.exists():
                    state = json.loads(path.read_text())
                    if state.get('status') == 'RUNNING' and state.get('control_address'):
                        break
                    if state.get('status') == 'FAILED':
                        self.fail(state)
                time.sleep(.05)
            else:
                self.fail('CPU guardian fixture did not become ready')
            result = subprocess.run(['bash', str(TOOLS / 'server_stop.sh'), str(session)],
                                    capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'STOPPED')
            self.assertEqual(child.wait(timeout=10), 0)
            final = json.loads((session / 'guardian_state.json').read_text())
            self.assertEqual(final['cleanup']['ownership'], 'DIRECT_POPEN_AND_HELD_BIRTH_PIDFD')
            self.assertEqual(final['cleanup']['signals'], ['SIGTERM'])
            self.assertFalse(final['pidfd_held'])
            (EVIDENCE / 'inert_child_cleanup_proof.json').write_text(json.dumps(final, indent=2))
        finally:
            if child.poll() is None:
                guardian.request(session / 'guardian_state.json', session / 'guardian.token', 'stop')
                child.wait(timeout=30)
        # Evidence is retained in this owned NFS session; no recursive deletion.


def main():
    global EVIDENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert args.evidence_dir.is_absolute() and args.evidence_dir.resolve().is_relative_to(Path('/nfs/baiyuntian/assets/model').resolve())
    args.evidence_dir.mkdir(exist_ok=False)
    EVIDENCE = args.evidence_dir
    os.chdir(EVIDENCE)
    import torch
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GuideTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert not torch.cuda.is_initialized()
    proof = {'status': 'CPU_GUIDE_API_TESTS_PASSED' if result.wasSuccessful() else 'FAILED',
             'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
             'skipped': len(result.skipped), 'cuda_initialized': torch.cuda.is_initialized(),
             'real_model_started': False, 'physical_robot': False,
             'scope': 'CPU SimulatedBackend with real loopback gRPC; fake8x2 chunks, not formal model performance'}
    (EVIDENCE / 'result.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
