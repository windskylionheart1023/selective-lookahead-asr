#!/usr/bin/env bash
# Canonical entry point for the streaming recipe.
#
# Pipeline:
#   1. Data + features            (--run_data): asr.sh stages 1-4, then the paper's
#      BPE model from local/bpe_unigram5000 (the training alignment counts its tokens).
#   2. Offline base training      (--run_offline): RoPE Conformer, full attention
#   3. Streaming fine-tune        (--run_streaming): asymmetric chunked masks,
#      causal XCQ depthwise conv, alignment-supervised early emission (SOFT).
#      Starts from the offline checkpoint (local/make_finetune_init.py) and reads
#      per-token alignments (local/alignment for train, a uniform placeholder for dev).
#   4. Streaming decode + scoring (--run_decode): chunked simulation, either
#      static lookahead (decode_mode=static, nrc=N future chunks) or dynamic
#      future chunks (decode_mode=dynamic: strategy D with the learned trigger,
#      or trigger=sr_cem|top1; commit-stable-prefix unless --csp false).
#
# Examples:
#   ./run.sh --run_data true
#   ./run.sh --run_offline true
#   ./run.sh --run_streaming true
#   ./run.sh --run_decode true --decode_mode static --nrc 1
#   ./run.sh --run_decode true --decode_mode dynamic --threshold 0.70

set -euo pipefail

train_set=train_lib360
valid_set=dev_lib360
test_sets=test_lib360
nbpe=5000

offline_config=conf/train_asr_conformer_offline_rope_newfrontend.yaml
streaming_config=conf/train_asr_SOFT_align_finetune_FROM_XCQDCCONV_R4MAX_BS32.yaml
inference_config=conf/decode_asr_bs20.yaml
inference_model=valid.acc.ave_10best.pth

run_data=false
run_offline=false
run_streaming=false
run_decode=false

# decode options
decode_mode=static      # static | dynamic
chunk_size=16           # encoder chunk size (frames after subsampling)
nrc=1                   # static: number of future (right) chunks of lookahead
trigger=wait_policy     # dynamic: wait_policy (learned) | sr_cem | top1
threshold=0.70          # dynamic: defer threshold (paper: learned 0.70/0.85, sr_cem 0.85/0.95, top1 0.95)
mfc=4                   # dynamic: max future chunks a deferral may accumulate
csp=true                # dynamic: commit-stable-prefix (false = whole-chunk rollback)
ngpu=1
stage_data_to_scratch=false  # copy audio + model to node-local scratch before decoding

. ./utils/parse_options.sh
. ./path.sh

common_args=(
    --lang en --nbpe "${nbpe}" --token_type bpe
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}"
    --bpe_train_text "data/${train_set}/text"
    --use_lm false --use_ngram false
)

if "${run_data}"; then
    ./asr.sh --stage 1 --stop_stage 4 --ngpu 0 "${common_args[@]}" \
        --asr_config "${offline_config}"
    mkdir -p data/en_token_list/bpe_unigram5000
    cp local/bpe_unigram5000/bpe.model local/bpe_unigram5000/tokens.txt \
        data/en_token_list/bpe_unigram5000/
fi

if "${run_offline}"; then
    ./asr.sh --stage 10 --stop_stage 11 --ngpu "${ngpu}" "${common_args[@]}" \
        --asr_config "${offline_config}"
fi

if "${run_streaming}"; then
    # Init = offline model without the decoder token embedding, as in the paper
    # (the output path is init_param in ${streaming_config}).
    offline_exp="exp/asr_$(basename "${offline_config}" .yaml)_raw_en_bpe${nbpe}"
    mkdir -p exp/ready_for_fine_tune
    python3 local/make_finetune_init.py "${offline_exp}/${inference_model}" \
        exp/ready_for_fine_tune/transferred_init_rope_newfrontend.pth
    # Per-token alignments for the early-emission losses.
    gunzip -c local/alignment/train_lib360.token_frame_mapping.gz \
        > "dump/raw/${train_set}/token_frame_mapping"
    python3 local/make_uniform_alignment.py "dump/raw/${valid_set}" \
        data/en_token_list/bpe_unigram5000/bpe.model
    ./asr.sh --stage 10 --stop_stage 11 --ngpu "${ngpu}" "${common_args[@]}" \
        --asr_config "${streaming_config}" \
        --use_alignment true --alignment_file token_frame_mapping
fi

if "${run_decode}"; then
    inf_args="--attn_edge_stop_margin 2 --unmask_xattn_at_final true"
    stream_nrc="${nrc}"
    tag="stream_${decode_mode}"
    case "${decode_mode}" in
        static)
            inf_args+=" --cross_attn_num_right_chunks ${nrc}"
            tag+="_nrc${nrc}" ;;
        dynamic)
            stream_nrc="${mfc}"
            inf_args+=" --dynamic_future_chunks true --dynamic_mode token_chunk_resume"
            case "${trigger}" in
                wait_policy) inf_args+=" --dynamic_signal_type wait_policy --sr_cem_ckpt local/triggers/wait_policy_A.pt --sr_cem_threshold ${threshold}" ;;
                sr_cem) inf_args+=" --dynamic_signal_type sr_cem_causal --sr_cem_ckpt local/triggers/sr_cem_A_stdz.pt --sr_cem_threshold ${threshold}" ;;
                top1) inf_args+=" --dynamic_signal_type top1_prob --dynamic_top1_prob_threshold ${threshold}" ;;
                *) echo "unknown trigger: ${trigger}" >&2; exit 2 ;;
            esac
            inf_args+=" --max_future_chunks ${mfc} --cross_attn_num_right_chunks ${mfc}"
            tag+="_${trigger}_t${threshold}_b${mfc}"
            if "${csp}"; then
                inf_args+=" --commit_stable_prefix true"
                tag+="_csp"
            fi ;;
        *) echo "unknown decode_mode: ${decode_mode}" >&2; exit 2 ;;
    esac

    ./asr_streaming_sim.sh --stage 12 --stop_stage 13 \
        --ngpu "${ngpu}" --gpu_inference true --inference_nj 1 \
        "${common_args[@]}" \
        --asr_config "${streaming_config}" \
        --inference_config "${inference_config}" \
        --inference_asr_model "${inference_model}" \
        --use_streaming_sim true --use_chunked_streaming true \
        --stream_chunk_size "${chunk_size}" \
        --stream_num_left_chunks -1 \
        --stream_num_right_chunks "${stream_nrc}" \
        --stream_num_global_tokens 0 \
        --use_decoder_self_kvcache true --use_decoder_cross_kvcache true \
        --use_encoder_self_kvcache false \
        --stage_data_to_scratch "${stage_data_to_scratch}" \
        --inference_args "${inf_args}" \
        --inference_tag "${tag}"
fi
