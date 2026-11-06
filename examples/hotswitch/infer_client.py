"""Inference-only client: returns a raw/processed chunk, sends no robot command."""
import argparse
import json
import math
import os
from pathlib import Path


def infer_once(address, task_id, observation, timeout=60, previous_session_id=None):
    from lerobot.async_inference.hotswitch.runtime import GrpcHotSwitchTransport, HotSwitchClientCore
    from lerobot.async_inference.hotswitch.simulation import SimulatedRobot

    # The explicit hold belongs to an in-memory simulator, never to hardware.
    simulated_control = SimulatedRobot()
    transport = GrpcHotSwitchTransport(address, timeout=timeout)
    core = HotSwitchClientCore(transport, safe_hold=simulated_control.hold)
    try:
        current = transport.status()
        if current.get('session_id') and not previous_session_id:
            raise RuntimeError('An existing session is present. End your previous client, then explicitly pass its --previous-session-id; do not take over another client.')
        core.connect(task_id, previous_session_id=previous_session_id)
        if not core.request_actions(observation):
            raise RuntimeError('The response was discarded by identity/alignment checks')
        with core.lock:
            pairs = list(core.queue)
            result = {'status': 'INFERENCE_CHUNK_RECEIVED', 'task_id': task_id,
                      'session_id': core.status['session_id'], 'policy_epoch': core.status['policy_epoch'],
                      'request_id': core.queue_identity['request_id'],
                      'prefix_steps': core.last_inference_alignment['prefix_steps'],
                      'raw_actions': [pair[0] for pair in pairs],
                      'processed_actions': [pair[1] for pair in pairs],
                      'chunk_length': len(pairs), 'action_dim': len(pairs[0][0]),
                      'physical_robot': False, 'robot_commands_sent': 0,
                      'scope': 'Bootstrap inference only. These are predictions, not accepted robot actions.'}
        return result
    finally:
        core.stop()
        simulated_control.hold()
        transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--task-id', default='A')
    parser.add_argument('--observation', required=True, type=Path, help='Raw observation safetensors; no Base or Adapter weights are loaded by this client')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--previous-session-id')
    parser.add_argument('--execute', action='store_true', help='Actually contact the existing service; otherwise print a read-only plan')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'PLAN_ONLY_NOT_EXECUTED', 'address': args.address,
                          'task_id': args.task_id, 'observation': str(args.observation),
                          'output': str(args.output), 'physical_robot': False}))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Set CUDA_VISIBLE_DEVICES="" for this CPU RPC client')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be positive')
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error('Use a new output file in an existing evidence directory')
    from safetensors.torch import load_file
    observation = load_file(str(args.observation), device='cpu')
    result = infer_once(args.address, args.task_id, observation, args.timeout, args.previous_session_id)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('raw_actions', 'processed_actions')}))
    print('First processed prediction:', result['processed_actions'][0])


if __name__ == '__main__':
    main()
