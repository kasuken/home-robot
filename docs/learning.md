# Learning new skills

`src/laika/learning.py` (the owner's ask). When someone asks for something none of Laika's tools can do, she says she cannot do it yet and offers to **try to learn it**. On a clear yes, a [GitHub Copilot SDK](https://github.com/github/copilot-sdk/tree/main/python) session edits her own repository (`[learning] repo_dir`, `/home/ema/home-robot` on the Pi). When it is done she tells you and **restarts herself** (`systemctl restart laika`) to load it.

## The flow

1. The model calls `propose_skill(description)` (one English sentence, no names) and, in the same reply, asks "vuoi che provi a imparare?".
2. Only in the **next** turn, after a clear yes, it calls `learn_skill()`. The code refuses it in the same turn as the proposal, when there is no fresh proposal (`[learning] proposal_ttl_s`), after web content in the turn, and in a proactive turn. A second job cannot start while one runs, and `max_per_day` caps the starts (kept across restarts in `learning/history.json`).
3. The job runs in a background thread; the conversation and the safety monitor are not blocked. Steps:
   - the clone must be clean (otherwise she says no); `git pull --ff-only`; a new branch `skill/<slug>-<time>`;
   - the Copilot session (`working_directory` = the clone, `[learning] model`, `timeout_s`). Its permission requests go through `permission_verdict`: no `git commit/push/checkout…`, `sudo`, `systemctl`, `gh`, `pip`, network, no reading of `secrets.toml`, no writes outside the clone or to a protected path;
   - Laika's own check of the result (`check_changes`): the diff against the commit the job started from (so a commit Copilot makes itself is judged too) must not touch the **protected paths** (safety, behaviour, hardware, robot, limits, `commands.py`, `actions.py`, config and the TOML files, storage, deploy, vendor snapshot, `AGENTS.md`, `pyproject.toml`, the learning code and its tests), `store.py`, `tools.py` and `app.py` may only grow (a new tool and its wiring, never an edit to the checks already there), and added source lines may not reach for the motors or the system (`picarx`, `.drive(`, `subprocess`, `sudo`, `systemctl`…). Wheels stay behind `SafetyController`;
   - the whole unit-test suite (`test_timeout_s`; Copilot itself is told to run only its own tests, the full suite is slow on a Pi) and an `import laika.app` check must pass;
   - then one commit on the `skill/` branch (`pyproject.toml` version +0.0.1), a fast-forward **merge into the branch the clone was on (main)**, tag `vX.Y.Z` on it, push of main and the tag, and `gh release create` (`push`, `release`). The `skill/` branch is deleted after the merge. If the clone was on a detached commit or the merge is not a fast-forward, the skill stays on its `skill/` branch (checked out) instead. Publishing is best effort: if it fails the skill stays committed locally and she says so.
   - any failure restores the clone (`checkout -f`, `clean -fd`, branch deleted) and she says "Non sono riuscita a imparare questa cosa".
4. Success: "Ho imparato una cosa nuova: <title>. Mi riavvio adesso per usarla: torno tra un momento." Then, after the last word, `systemctl restart laika` runs detached (she is root under systemd, so no `sudo`). `[learning] restart = false`, or a run outside systemd (`--sim`, a terminal: no `NOTIFY_SOCKET`), or a failed restart, make her say instead: "lancia il comando sudo systemctl restart laika".

The skill is on `main` before the restart, so the restart loads it from there. To undo one: `git revert <its commit>` (or `git reset --hard <the previous tag>`) in the clone, then `sudo systemctl restart laika`.

## Setting up on the Pi

```sh
pip install github-copilot-sdk        # Python 3.11+; the runtime is fetched on first use
sudo apt install gh                   # the GitHub CLI, for the release (not on Raspberry Pi OS by default)
```

Laika runs as root under systemd, so a login in `ema`'s home is not seen. Give her a token instead, in `config/secrets.toml` (`[learning] github_token = "..."`) or `LAIKA_GITHUB_TOKEN` in `/etc/laika/laika.env`. It needs Copilot access ("Copilot Requests") and, for push and release, write access to the repository's contents. It reaches git only through environment variables (never an argument or the log), and `gh` as `GH_TOKEN`. Git runs with `safe.directory` set for the clone, because root does not own `/home/ema`. Files Copilot creates are owned by root: `sudo chown -R ema: /home/ema/home-robot` if that bothers you.

## Limits to know

- The permission filter on Copilot's commands is best effort. Everything Copilot runs, and the tests the gate runs, executes as root on the robot. The real gate is the diff check, which cannot see what a command did outside the clone.
- If the service stops or the watchdog kills it in the middle of a job (before the merge), the clone stays on the `skill/` branch with half-written files, and the next start imports them (a crash loop ends at `StartLimitBurst`). Recover with `git checkout -f main && git clean -fd` in the clone, then `sudo systemctl reset-failed laika && sudo systemctl start laika`.
- The automatic restart loads code that passed the unit tests and the import check, but not a run on real hardware. If the new skill still breaks start-up, systemd stops after `StartLimitBurst` failures; recover as above.

## Not verified yet

Written without access to the robot and with no network to Copilot: the SDK glue (`_copilot_session`) follows the SDK's documented client/session/event API but has not run against it. Unknown until tried on the Pi: whether the Copilot runtime fits in 1 GB of RAM next to Laika, how long a session takes, and the quality of what it writes. The rest (flow, gates, restore, publishing order) is covered by `tests/learning/` with a fake runner and fake git.
