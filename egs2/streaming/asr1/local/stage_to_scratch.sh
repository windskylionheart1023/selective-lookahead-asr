#!/usr/bin/env bash
# Stage data or model files to node-local scratch for one job.
#
# The cluster's NFS shares (audioslave in particular) are unstable; copying
# what a job reads to the execute node's local disk first makes the job
# immune to NFS hiccups after staging. Under Condor the per-job scratch dir
# (_CONDOR_SCRATCH_DIR) is used and cleaned automatically on job exit;
# otherwise TMPDIR (or /tmp) is used and the CALLER should remove the
# printed directory afterwards.
#
# Usage:
#   staged_dir=$(local/stage_to_scratch.sh <data_dir>)
#       Copies the Kaldi-style data dir metadata; if wav.scp holds plain
#       file paths, copies every referenced audio file and rewrites
#       wav.scp to the local copies.
#   staged_file=$(local/stage_to_scratch.sh --file <path>)
#       Copies one file (e.g. a model checkpoint).
#
# On ANY failure (no space, unreadable source, piped wav.scp entries) it
# prints the ORIGINAL path and exits 0, so callers can always use the
# printed path: staging degrades gracefully to plain NFS access.
# All copies retry 3 times (transient NFS errors like ESTALE).

set -u

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') (stage_to_scratch) $*" >&2; }

_copy_retry() { # src dst
    local i
    for i in 1 2 3; do
        if cp "$1" "$2" 2>/dev/null; then return 0; fi
        log "copy attempt ${i}/3 failed for '$1', retrying..."
        sleep $((i * 2))
    done
    return 1
}

scratch_base="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
# STAGE_ROOT lets a caller group several stage calls under one directory it
# can clean up afterwards (outside Condor, where nothing auto-cleans).
stage_root="${STAGE_ROOT:-${scratch_base}/espnet_stage_$$}"

if [ "${1:-}" = "--file" ]; then
    src="${2:?usage: stage_to_scratch.sh --file <path>}"
    if [ ! -f "${src}" ]; then log "not a file: '${src}'; using original"; echo "${src}"; exit 0; fi
    mkdir -p "${stage_root}/files" 2>/dev/null || { echo "${src}"; exit 0; }
    dst="${stage_root}/files/$(basename "${src}")"
    if _copy_retry "${src}" "${dst}"; then
        log "staged file '${src}' -> '${dst}'"
        echo "${dst}"
    else
        log "staging failed for '${src}'; using original"
        rm -f "${dst}" 2>/dev/null
        echo "${src}"
    fi
    exit 0
fi

src_dir="${1:?usage: stage_to_scratch.sh <data_dir>}"
if [ ! -d "${src_dir}" ]; then log "not a dir: '${src_dir}'; using original"; echo "${src_dir}"; exit 0; fi
dst_dir="${stage_root}/$(basename "${src_dir}")"
if ! mkdir -p "${dst_dir}/audio"; then log "cannot create '${dst_dir}'; using original"; echo "${src_dir}"; exit 0; fi

fallback() {
    log "$1; using original '${src_dir}'"
    rm -rf "${stage_root}" 2>/dev/null
    echo "${src_dir}"
    exit 0
}

# 1. Metadata (small files: text, utt2spk, feats_type, ...)
for f in "${src_dir}"/*; do
    [ -f "$f" ] || continue
    _copy_retry "$f" "${dst_dir}/$(basename "$f")" || fallback "metadata copy failed for '$f'"
done

# 2. Audio referenced by wav.scp (plain paths only; command pipes are left
#    on NFS since their inputs cannot be relocated generically).
if [ -f "${src_dir}/wav.scp" ]; then
    if grep -q "|" "${src_dir}/wav.scp"; then
        log "wav.scp contains command pipes; audio stays on its original storage"
    else
        # Space check with 20% headroom.
        need=$(awk '{print $2}' "${src_dir}/wav.scp" | xargs -d '\n' du -Lcb 2>/dev/null | tail -1 | cut -f1)
        avail=$(df -B1 --output=avail "${dst_dir}" | tail -1 | tr -d ' ')
        if [ -z "${need}" ] || [ "${avail}" -lt $((need + need / 5)) ]; then
            fallback "insufficient scratch space (need ~${need:-?} B, avail ${avail} B)"
        fi
        : > "${dst_dir}/wav.scp"
        while read -r uttid path; do
            ext="${path##*.}"
            dst_audio="${dst_dir}/audio/${uttid}.${ext}"
            _copy_retry "${path}" "${dst_audio}" || fallback "audio copy failed for '${path}'"
            echo "${uttid} ${dst_audio}" >> "${dst_dir}/wav.scp"
        done < "${src_dir}/wav.scp"
        log "staged $(wc -l < "${dst_dir}/wav.scp") audio files to '${dst_dir}/audio'"
    fi
fi

log "staged data dir '${src_dir}' -> '${dst_dir}'"
echo "${dst_dir}"
