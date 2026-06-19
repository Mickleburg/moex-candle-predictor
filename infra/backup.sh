#!/usr/bin/env bash
# Backup the agent state-store + cycle results + shadow log. Idempotent, prune-old.
# Uses sqlite's online .backup so a backup taken while the agent is running is consistent.
#
#   ./infra/backup.sh                 # -> data/agent/backups/<UTC-timestamp>/
#   BACKUP_KEEP=30 ./infra/backup.sh  # keep the most recent 30 (default 14)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="${REPO_ROOT}/data/agent"
STATE_DB="${AGENT_DIR}/state.sqlite"
DEST_ROOT="${AGENT_DIR}/backups"
KEEP="${BACKUP_KEEP:-14}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${DEST_ROOT}/${STAMP}"

mkdir -p "${DEST}"

if [ -f "${STATE_DB}" ]; then
  # consistent hot backup of the live DB
  sqlite3 "${STATE_DB}" ".backup '${DEST}/state.sqlite'" 2>/dev/null \
    || cp "${STATE_DB}" "${DEST}/state.sqlite"
fi
[ -d "${AGENT_DIR}/cycles" ] && tar -czf "${DEST}/cycles.tar.gz" -C "${AGENT_DIR}" cycles || true
[ -f "${AGENT_DIR}/shadow_pnl.jsonl" ] && cp "${AGENT_DIR}/shadow_pnl.jsonl" "${DEST}/" || true

echo "backup written to ${DEST}"

# prune: keep the newest ${KEEP} timestamped dirs
mapfile -t old < <(ls -1dt "${DEST_ROOT}"/*/ 2>/dev/null | tail -n +$((KEEP + 1)) || true)
for d in "${old[@]:-}"; do [ -n "${d}" ] && rm -rf "${d}" && echo "pruned ${d}"; done
