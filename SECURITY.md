# Security

Report security bugs by email to security@lumbridgecorp.com, or use "Report a vulnerability" on this repo's Security tab. Do not open a public issue with exploit details.

Include your `lumbridge-computer --version`, OS and hardware, whether the problem is local or remote, steps to reproduce, and the impact you expect. Remove secrets from logs and examples. A small reproduction helps.

The latest release and the latest `main` are supported. We answer in the same email thread. There is no fixed response deadline.

## Trust model

- **The model registry is trusted local configuration.** It holds the commands that start models, so whoever can write `registry/models.yaml` can run code as the agent's user. Keep it in your own repository and review changes to it.
- **Scenes and eval suites are untrusted, shareable data.** They reference model ids from your registry and set parameters; they must never be able to run a command. A Scene that names an id you don't have is refused. A way for a Scene or eval suite to start a process or run a command is a security bug.
- **Downloaded weights** must be checksum-verified before they go into the trusted registry.
- **Secrets** belong in the environment or an OS-managed secret store, never in manifests, logs or result files. `doctor` reads the Supabase `.env` for key names only and never reads container environments.
- **Network.** The gateway (`:8011`) and the read-only API have no authentication unless you set `api --token`. They are meant for one machine or a private network. vLLM engines with sleep enabled run in dev mode bound to 127.0.0.1; the gateway refuses their engine-control routes. A way around that is in scope.
- **Local users.** Anyone with the agent's user account controls the models. That is out of scope.
