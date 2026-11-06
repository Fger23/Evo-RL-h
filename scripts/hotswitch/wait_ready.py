"""Wait for the new session's real warm loopback service; never starts a model."""
import argparse
import json
import math
from pathlib import Path
import time


def wait_ready(session, timeout=900):
    from grpc import RpcError
    from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport

    session = Path(session).resolve(strict=True)
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            state_path = session / 'guardian_state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get('status') in ('FAILED', 'STOPPED', 'EXITED'):
                    raise RuntimeError(f'Service stopped before ready: {state.get("error", state["status"])}; see {session}/server.stderr.log')
            ready = json.loads((session / 'ready.json').read_text())
            address = ready['address']
            assert address.startswith('127.0.0.1:')
            transport = GrpcHotSwitchTransport(address, timeout=2)
            try:
                live = transport.status()
                assert {'A', 'A_copy'}.issubset(live['ready']) and not live['stopped']
                return address
            finally:
                transport.close()
        except (FileNotFoundError, json.JSONDecodeError, OSError, AssertionError, RpcError) as error:
            last_error = repr(error)
        time.sleep(.5)
    raise TimeoutError(f'Service did not become ready within {timeout}s: {last_error}; see {session}/server.stderr.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive')
    print(wait_ready(args.session, args.timeout))


if __name__ == '__main__':
    main()
