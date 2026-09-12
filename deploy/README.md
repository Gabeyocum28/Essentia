# Deploy

The Essentia API and worker as a Docker Compose stack, meant to run on the VM
behind a shared Caddy instance. Nothing here binds a host port — Caddy reaches
the containers by name (`essentia-api`) over the external `web` Docker
network, using the site block in `Caddyfile.essentia`.

## Commands

First-time setup (and to pick up new commits later):

```bash
bash deploy/bootstrap.sh
```

This clones/updates the repo at `/home/ubuntu/stacks/essentia`, creates
`deploy/.env` from `deploy/.env.example` if it doesn't exist yet, appends
`Caddyfile.essentia` to the shared Caddyfile (backing it up first) the first
time it hasn't been added, and brings the stack up.

Tail the worker's logs:

```bash
docker compose -f deploy/docker-compose.yml logs -f worker
```

Redeploy after pulling new commits:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

## Secrets

Real config lives in `deploy/.env` (copied from `.env.example`, then filled
in with `MONGODB_URI`). It is never committed — it's gitignored and
dockerignored — and `bootstrap.sh` never overwrites an existing one.

## Caddy

`/home/ubuntu/stacks/caddy` is shared with other sites on the same VM. We own
only the `essentia.gabeyocum.com { ... }` block appended from
`deploy/Caddyfile.essentia`; do not touch the rest of that Caddyfile.
