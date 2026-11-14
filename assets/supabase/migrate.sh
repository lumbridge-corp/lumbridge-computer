#!/bin/sh
# Apply SQL migrations to the box's Supabase database, in filename order,
# each in its own transaction together with its history row. Already-recorded
# versions are skipped, so this is safe to re-run.
#
#   sh migrate.sh --name home <migrations-dir>
#
# History uses the Supabase CLI's table (supabase_migrations.schema_migrations), so
# `supabase migration list` against this database agrees with what ran here.
set -eu

name=""
while [ $# -gt 0 ]; do
    case "$1" in
        --name) name="$2"; shift 2 ;;
        -h|--help) sed -n 2,9p "$0"; exit 0 ;;
        *) break ;;
    esac
done
src="${1:-}"
[ -n "$name" ] && [ -d "$src" ] || { echo "usage: sh migrate.sh --name <name> <migrations-dir>" >&2; exit 2; }
src=$(cd "$src" && pwd)
cd "${LUMBRIDGE_COMPUTER_HOME:-$HOME/lumbridge-computer}/supabase-$name"

psql() { docker compose exec -T db psql -U supabase_admin -d postgres -v ON_ERROR_STOP=1 -qAt "$@"; }

psql -c "create schema if not exists supabase_migrations;
         create table if not exists supabase_migrations.schema_migrations
           (version text primary key, statements text[], name text);" >/dev/null

applied=0
for f in "$src"/*.sql; do
    base=$(basename "$f" .sql)
    version=${base%%_*}
    label=${base#*_}
    # Several repos ship migrations into one Server, and their timestamps can collide
    # (lumbridge-os and lumbridge-mcp both have 20260927130000). A migration is the
    # same only if version and name both match; a colliding one is recorded under its
    # full file name so it still has a unique history row.
    recorded=$(psql -c "select name from supabase_migrations.schema_migrations where version in ('$version', '$base')")
    case "$recorded" in
        *"$label"*) continue ;;
        '') ;;
        *) version=$base ;;
    esac
    if [ "$(psql -c "select 1 from supabase_migrations.schema_migrations where version = '$version'")" = 1 ]; then
        continue
    fi
    echo "apply $base"
    record="insert into supabase_migrations.schema_migrations (version, name) values ('$version', '$label');"
    if grep -Eiq '^[[:space:]]*(begin|start transaction)[[:space:]]*;' "$f"; then
        # The file commits itself; wrapping it would split it at its own COMMIT.
        # ON_ERROR_STOP means a failed file never reaches the history row.
        psql -f - < "$f" >/dev/null
        psql -c "$record" >/dev/null
    else
        { cat "$f"; printf '\n%s\n' "$record"; } | psql --single-transaction >/dev/null
    fi
    applied=$((applied + 1))
done
echo "applied $applied migration(s); $(psql -c 'select count(*) from supabase_migrations.schema_migrations') recorded"
