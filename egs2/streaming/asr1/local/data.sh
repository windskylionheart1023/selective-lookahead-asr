#!/usr/bin/env bash
# LibriSpeech data preparation for the streaming recipe.
#
# Wraps the standard egs2/librispeech data preparation and links the subsets
# under the names this recipe's configs use:
#   train_lib360 = train-clean-360
#   dev_lib360   = dev-clean
#   test_lib360  = test-clean
#
# Set LIBRISPEECH= in db.sh to an existing corpus root to skip the download.

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=2

. ./utils/parse_options.sh
. ./db.sh
. ./path.sh
. ./cmd.sh

if [ -z "${LIBRISPEECH:-}" ]; then
    log "Error: set LIBRISPEECH= in db.sh (corpus root, or 'downloads' to fetch)."
    exit 1
fi

data_url=www.openslr.org/resources/12
train_part="train-clean-360"
dev_part="dev-clean"
test_part="test-clean"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "${LIBRISPEECH}" = "downloads" ] || [ ! -d "${LIBRISPEECH}/LibriSpeech" ]; then
        log "Stage 1: downloading LibriSpeech parts to '${LIBRISPEECH}'"
        mkdir -p "${LIBRISPEECH}"
        for part in ${train_part} ${dev_part} ${test_part}; do
            ../../librispeech/asr1/local/download_and_untar.sh \
                "${LIBRISPEECH}" "${data_url}" "${part}"
        done
    else
        log "Stage 1: using existing corpus at ${LIBRISPEECH}"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: preparing Kaldi-style data directories"
    for part in ${train_part} ${dev_part} ${test_part}; do
        ../../librispeech/asr1/local/data_prep.sh \
            "${LIBRISPEECH}/LibriSpeech/${part}" "data/${part//-/_}"
    done
    # Recipe-canonical names
    utils/copy_data_dir.sh data/train_clean_360 data/train_lib360
    utils/copy_data_dir.sh data/dev_clean data/dev_lib360
    utils/copy_data_dir.sh data/test_clean data/test_lib360
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
