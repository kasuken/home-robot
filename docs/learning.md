# Learning new skills

`src/laika/learning.py` (the owner's ask). When someone asks for something none of Laika's tools can do, she says she cannot do it yet and offers to **try to learn it**. On a clear yes, a [GitHub Copilot SDK](https://github.com/github/copilot-sdk/tree/main/python) session edits her own repository (`[learning] repo_dir`, `/home/ema/home-robot` on the Pi). When it is done she tells you, and asks you to restart her with `sudo systemctl restart laika`. She never restarts herself.

## The flow

1. The model calls `propose_skill(description)` (one English sentence, no names) and, in the same reply, asks "vuoi che provi a imparare?".
2. Only in the **next** turn, after a clear yes, it calls `learn_skill()`. The code refuses it in the same turn as the proposal, when there is no fresh proposal (`[learning] proposal_ttl_s`), after web content in the turn, and in a proactive turn. A second job cannot start while one runs, and `max_per_day` caps the starts (kept across restarts in `learning/history.json`).
3. The job runs in a background thread; the conversation and the safety monitor are not blocked. Steps:
   - the clone must be clean (otherwise she says no); `git pull --ff-only`; a new branch `skill/<slug>-<time>`;
   - the Copilot session (`working_directory` = the clone, `[learning] model`, `timeout_s`). Its permission requests go through `permission_verdict`: no `git commit/push/checkout…`, `sudo`, `systemctl`, `gh`, `pip`, network, no reading of `secrets.toml`, no writes outside the clone or to a protected path;
   - Laika's own check of the result (`check_changes`): the diff must not touch the **protected paths** (safety, behaviour, hardware, robot, limits, config and the TOML files, storage, deploy, vendor snapshot, `AGENTS.md`, `pyproject.toml`, the learning code and its tests), `store.py` may only grow, and added source lines may not reach for the motors or the system (`picarx`, `.drive(`, `subprocess`, `sudo`, `systemctl`…). Wheels stay behind `SafetyController`;
   - the whole unit-test suite (`test_timeout_s`) and an `import laika.app` check must pass;
   - then one commit (`pyproject.toml` version +0.0.1), tag `vX.Y.Z`, push of the branch and the tag, and `gh release create` (`push`, `release`). Publishing is best effort: if it fails the skill stays committed locally and she says so.
   - any failure restores the clone (`checkout -f`, `clean -fd`, branch deleted) and she says "Non sono riuscita a imparare questa cosa".
4. Success: "Ho imparato una cosa nuova: <title>. Per usarla devo riavviarmi: lancia il comando sudo systemctl restart laika."

The new skill lives on the `skill/…` branch, which stays checked out so the restart loads it. To go back: `git checkout main && sudo systemctl restart laika`; to keep it, merge the branch (or the release's tag) into `main`.

## Setting up on the Pi

```sh
pip install github-copilot-sdk        # Python 3.11+; the runtime is fetched on first use
```

Laika runs as root under systemd, so a login in `ema`'s home is not seen. Give her a token instead, in `config/secrets.toml` (`[learning] github_token = "..."`) or `LAIKA_GITHUB_TOKEN` in `/etc/laika/laika.env`. It needs Copilot access ("Copilot Requests") and, for push and release, write access to the repository's contents. It reaches git only through environment variables (never an argument or the log), and `gh` as `GH_TOKEN`. Git runs with `safe.directory` set for the clone, because root does not own `/home/ema`. Files Copilot creates are owned by root: `sudo chown -R ema: /home/ema/home-robot` if that bothers you.

## Not verified yet

Written without access to the robot and with no network to Copilot: the SDK glue (`_copilot_session`) follows the SDK's documented client/session/event API but has not run against it. Unknown until tried on the Pi: whether the Copilot runtime fits in 1 GB of RAM next to Laika, how long a session takes, and the quality of what it writes. The rest (flow, gates, restore, publishing order) is covered by `tests/learning/` with a fake runner and fake git.
