# AGENTS.md

Conventions agents must follow when working in this repository.

## Deployment workflow (always use git, never scp)

The production host `brandon@192.168.1.195` runs this code from
`/opt/speciesnet/animaltracker`, which is a clone of the same `origin`
(`github.com/bpasker/animaltracker`, branch `main`). Both ends use
passwordless SSH key auth and `NOPASSWD` sudo for `brandon`.

After ANY source change in `src/`, `config/`, or `systemd/`:

1. **Locally**: stage and commit the change with a clear message, then push.
   ```bash
   git add -A && git commit -m "<concise message>" && git push origin main
   ```
2. **Remote**: pull on the Jetson and restart affected services.
   ```bash
   ssh brandon@192.168.1.195 'cd /opt/speciesnet/animaltracker && \
     sudo git pull --ff-only origin main && \
     sudo systemctl restart detector@cam1 detector@cam2 && \
     sudo systemctl is-active detector@cam1 detector@cam2'
   ```

Do **not** `scp` files to the remote or hand-edit
`/opt/speciesnet/animaltracker/...` directly. Treat the remote checkout as
read-only except via `git pull`. If a hotfix was applied via scp in the past,
revert any `*.bak` files and reconcile to git before continuing.

## Notes

- Restart only the services impacted by the change. Most code changes affect
  both `detector@cam1` and `detector@cam2`; pure config edits also need a
  restart.
- For config-only edits to `/opt/speciesnet/animaltracker/config/cameras.yml`
  done directly on the remote (e.g. via the Web UI), commit the result back to
  git on the remote and push, so local stays in sync.
- Never commit secrets. `config/secrets.env` is gitignored; if it isn't,
  add it before committing.
