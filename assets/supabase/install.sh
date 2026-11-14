#!/bin/sh
# Install the box's Supabase: upstream self-hosted Supabase (pinned in UPSTREAM)
# plus lumbridge-computer.override.yml, as compose project lumbridge-computer-supabase-<name>
# in $LUMBRIDGE_COMPUTER_HOME/supabase-<name> (default ~/lumbridge-computer). Containers
# go in the cgroup slice $LUMBRIDGE_COMPUTER_SLICE (lumbridge-computer.slice).
#
# Runs on the box. Idempotent for everything except secrets: an existing .env is
# never regenerated, because rotating JWT/Postgres secrets under a live box logs out
# every device.
#
#   sh install.sh --name home --public-url https://box1.tail0000.ts.net:8443 [--serve]
#
# --serve  publish the API gateway on the tailnet: tailscale serve --https=<port> -> 127.0.0.1:8000.
#          tailscale serve only allows HTTPS on 443, 8443 and 10000.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
name=""
public_url=""
serve=0
slice=${LUMBRIDGE_COMPUTER_SLICE:-lumbridge-computer.slice}

while [ $# -gt 0 ]; do
    case "$1" in
        --name) name="$2"; shift 2 ;;
        --public-url) public_url="$2"; shift 2 ;;
        --serve) serve=1; shift ;;
        -h|--help) sed -n 2,15p "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$name" in
    ''|*[!a-z0-9-]*) echo "--name is required: lowercase letters, digits, hyphens" >&2; exit 2 ;;
esac
case "$slice" in
    ''|*[!A-Za-z0-9_.-]*) echo "LUMBRIDGE_COMPUTER_SLICE must be a systemd slice name like lumbridge-computer.slice" >&2; exit 2 ;;
    *.slice) ;;
    *) echo "LUMBRIDGE_COMPUTER_SLICE must end in .slice" >&2; exit 2 ;;
esac
case "$public_url" in
    https://*) ;;
    *) echo "--public-url is required and must be https:// (the tailnet URL devices use)" >&2; exit 2 ;;
esac

. "$here/UPSTREAM"
project="lumbridge-computer-supabase-$name"
dir="${LUMBRIDGE_COMPUTER_HOME:-$HOME/lumbridge-computer}/supabase-$name"
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
docker compose version >/dev/null || { echo "docker compose v2 is required" >&2; exit 1; }
command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 1; }

# 1. Upstream docker/ at the pinned commit.
if [ ! -f "$dir/docker-compose.yml" ]; then
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git clone -q --depth 1 --branch "$ref" --filter=blob:none --sparse "$repo" "$tmp/src"
    git -C "$tmp/src" sparse-checkout set docker
    got=$(git -C "$tmp/src" rev-parse HEAD)
    if [ "$got" != "$commit" ]; then
        echo "upstream $ref resolved to $got, expected $commit; refusing (tag moved?)" >&2
        exit 1
    fi
    mkdir -p "$dir"
    cp -R "$tmp/src/docker/." "$dir/"
    echo "$ref $commit" > "$dir/.lumbridge-computer-upstream"
fi
cp "$here/lumbridge-computer.override.yml" "$dir/lumbridge-computer.override.yml"

# 2. Configuration and secrets, written once.
cd "$dir"
if [ ! -f .env ]; then
    umask 077
    cp .env.example .env
    sh utils/generate-keys.sh --update-env >/dev/null
    sh utils/add-new-auth-keys.sh --update-env >/dev/null
    rm -f .env.old
    tenant="lumbridge-computer-$name"
    sed -i \
        -e "s|^COMPOSE_FILE=.*$|COMPOSE_FILE=docker-compose.yml:lumbridge-computer.override.yml|" \
        -e "s|^SUPABASE_PUBLIC_URL=.*$|SUPABASE_PUBLIC_URL=$public_url|" \
        -e "s|^API_EXTERNAL_URL=.*$|API_EXTERNAL_URL=$public_url/auth/v1|" \
        -e "s|^SITE_URL=.*$|SITE_URL=$public_url|" \
        -e "s|^DISABLE_SIGNUP=.*$|DISABLE_SIGNUP=true|" \
        -e "s|^ENABLE_PHONE_SIGNUP=.*$|ENABLE_PHONE_SIGNUP=false|" \
        -e "s|^ENABLE_PHONE_AUTOCONFIRM=.*$|ENABLE_PHONE_AUTOCONFIRM=false|" \
        -e "s|^POOLER_TENANT_ID=.*$|POOLER_TENANT_ID=$tenant|" \
        -e "s|^STUDIO_DEFAULT_ORGANIZATION=.*$|STUDIO_DEFAULT_ORGANIZATION=Lumbridge Computer|" \
        -e "s|^STUDIO_DEFAULT_PROJECT=.*$|STUDIO_DEFAULT_PROJECT=$name|" \
        -e "s|^OPENAI_API_KEY=.*$|OPENAI_API_KEY=|" \
        .env
    printf '\nCOMPOSE_PROJECT_NAME=%s\nLUMBRIDGE_COMPUTER_SLICE=%s\n' "$project" "$slice" >> .env
    # Upstream placeholders that must never survive into a running box.
    if grep -Eq 'your-super-secret|this_password_is_insecure|your-32-character|your-encryption-key' .env; then
        echo "secret generation left an upstream placeholder in $dir/.env; stopping" >&2
        exit 1
    fi
    echo "wrote $dir/.env (0600); secrets were generated and not printed"
else
    echo "keeping existing $dir/.env"
    # The overlay reads the slice from .env; an older .env without it falls back to
    # lumbridge-computer.slice, which moves the stack out of the slice Lumbridge Computer reads.
    grep -q '^LUMBRIDGE_COMPUTER_SLICE=' .env ||
        echo "note: $dir/.env has no LUMBRIDGE_COMPUTER_SLICE; containers will use lumbridge-computer.slice" >&2
fi
chmod 600 .env

# 3. Up.
docker compose pull -q
docker compose up -d --wait --wait-timeout 300
docker compose ps --format 'table {{.Service}}\t{{.Status}}'
# Older compose releases append to `ports:` instead of honouring `!override`, which
# would publish Postgres and the gateway on every interface. Refuse to leave that up.
if docker compose ps --format '{{.Ports}}' | grep -Eq '(0\.0\.0\.0|\[::\]):'; then
    docker compose down
    echo "a port was published beyond loopback (docker compose too old for !override?); stopped" >&2
    exit 1
fi

# 4. Tailnet.
if [ "$serve" = 1 ]; then
    port=${public_url##*:}
    case "$port" in 443|8443|10000) ;; *) port=443 ;; esac
    tailscale serve --bg --https="$port" http://127.0.0.1:8000
    echo "tailnet: $public_url -> 127.0.0.1:8000"
fi

echo "installed $project in $dir"
