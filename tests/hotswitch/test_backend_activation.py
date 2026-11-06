import unittest

from lerobot.async_inference.hotswitch.backend import SharedAdapterBackend


class AdapterModel:
    def __init__(self, broken=False):
        self.active_adapters = ["A"]
        self.broken = broken

    def set_adapter(self, name):
        if not self.broken:
            self.active_adapters = [name]

    def eval(self):
        return self

    def parameters(self):
        return iter(())


class ActivationTests(unittest.TestCase):
    def make_backend(self, broken=False):
        backend = SharedAdapterBackend.__new__(SharedAdapterBackend)
        backend.task_ids = ("A", "A_copy")
        backend.models = {"shared": AdapterModel(broken)}
        backend.active_task = "A"
        backend._dual = False
        return backend

    def test_task_identity_follows_the_actual_peft_selection(self):
        backend = self.make_backend()
        backend.activate("A_copy")
        self.assertEqual(backend.active_task, "A_copy")
        self.assertEqual(backend.policy.active_adapters, ["A_copy"])

    def test_silent_adapter_selection_failure_does_not_commit_task_identity(self):
        backend = self.make_backend(broken=True)
        with self.assertRaisesRegex(RuntimeError, "did not activate"):
            backend.activate("A_copy")
        self.assertEqual(backend.active_task, "A")
