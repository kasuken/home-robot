# SunFounder PiCar-X reference snapshot

The [`picar-x/`](picar-x/) folder contains an unchanged copy of the upstream repository's tracked files. It is included for reading the library and examples alongside Laika's code. A normal clone of `home-robot` includes the snapshot; no submodule setup is required.

## Provenance

| Field | Value |
| --- | --- |
| Upstream repository | [sunfounder/picar-x](https://github.com/sunfounder/picar-x) |
| Reference branch | `2.1.x` |
| Pinned commit | [`897698daf2719d830a5fb0b83951704bfee07d0c`](https://github.com/sunfounder/picar-x/tree/897698daf2719d830a5fb0b83951704bfee07d0c) |
| Imported | 15 October 2026 |
| Scope | All 58 tracked files at that commit, including media and hidden files |
| Verification | Every imported file's Git blob hash matched the upstream tree |
| Upstream licence | [LICENSE](picar-x/LICENSE), preserved unchanged |

The snapshot contains no upstream Git history or nested `.git` directory. Its version does not establish which version is installed on the physical robot.

## Useful starting points

- [Library implementation](picar-x/picarx/picarx.py).
- [Movement example](picar-x/example/2.move.py).
- [Obstacle avoidance](picar-x/example/4.avoiding_obstacles.py).
- [Cliff detection](picar-x/example/5.cliff_detection.py).
- [Camera and vision](picar-x/example/7.computer_vision.py).
- [Camera tracking](picar-x/example/8.stare_at_you.py).
- [Sound and music](picar-x/example/13.sound_background_music.py).
- [GPT examples](picar-x/gpt_examples/README.md).
- [Upstream README](picar-x/README.md).

The robot is already installed and verified with the vendor examples. This snapshot provides reference material for Laika's adapter; importing it does not install packages or run hardware commands.

## Maintenance

Keep upstream files unchanged and put Laika-specific code outside `picar-x/`. To refresh the reference, select an upstream commit, review its changes, replace the snapshot with that commit's tracked files, verify their blob hashes, and update the provenance table in the same commit. Preserve the upstream licence and notices with every refresh.
