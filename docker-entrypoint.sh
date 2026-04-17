#!/bin/sh
set -eu

is_truthy() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|on|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

repair_log_permissions() {
    log_dir="${LOG_DIR:-/workspace/logs}"
    log_file_name="${LOG_FILE_NAME:-chat-agent.log}"

    mkdir -p "$log_dir" || return 0
    touch "$log_dir/$log_file_name" 2>/dev/null || true
    chown -R app:app "$log_dir" 2>/dev/null || true
}

if [ "$(id -u)" -eq 0 ]; then
    if is_truthy "${LOG_TO_FILE:-true}"; then
        repair_log_permissions
    fi

    exec gosu app "$@"
fi

exec "$@"
