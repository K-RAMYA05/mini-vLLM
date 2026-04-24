#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/summarize_logs.sh <run_dir>

Example:
  bash scripts/summarize_logs.sh logs/h100/20260424_153000
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

RUN_DIR=$1

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

trim() {
  sed 's/^ *//; s/ *$//'
}

extract_first_match() {
  local pattern=$1
  local file=$2
  grep -m1 "$pattern" "$file" 2>/dev/null || true
}

extract_kernel_value() {
  local label=$1
  local file=$2
  extract_first_match "$label" "$file" | awk -F':' '{print $2}' | trim
}

extract_tok_s_arrow() {
  local prefix=$1
  local file=$2
  extract_first_match "$prefix" "$file" | sed -E 's/.*-> ([0-9.]+ tok\/s).*/\1/' | trim
}

extract_speedup() {
  local file=$1
  extract_first_match "Speedup:" "$file" | sed -E 's/.*Speedup:[[:space:]]*([0-9.]+x).*/\1/' | trim
}

extract_stress_field() {
  local field=$1
  local file=$2
  grep "stress_done" "$file" 2>/dev/null | tail -n 1 | sed -E "s/.*'$field': ([0-9.]+).*/\1/" | trim
}

extract_spec_field() {
  local pattern=$1
  local file=$2
  extract_first_match "$pattern" "$file" | sed -E 's/.*([0-9.]+x?).*/\1/' | trim
}

print_section() {
  echo
  echo "## $1"
}

print_kernel_table() {
  local dir=$1
  local found=0
  for file in "$dir"/kernel/*.log; do
    [[ -f "$file" ]] || continue
    if [[ $found -eq 0 ]]; then
      print_section "Kernel"
      echo "| Case | Triton paged | SDPA+gather | Speedup |"
      echo "|---|---:|---:|---:|"
      found=1
    fi
    local case_name
    case_name=$(basename "$file" .log)
    local triton_ms
    triton_ms=$(extract_kernel_value "Triton paged" "$file")
    local sdpa_ms
    sdpa_ms=$(extract_kernel_value "SDPA+gather" "$file")
    local speedup
    speedup=$(extract_speedup "$file")
    echo "| $case_name | $triton_ms | $sdpa_ms | $speedup |"
  done
}

print_throughput_group() {
  local title=$1
  local glob_pattern=$2
  local found=0
  for file in $glob_pattern; do
    [[ -f "$file" ]] || continue
    if [[ $found -eq 0 ]]; then
      print_section "$title"
      echo "| Run | mini-vLLM | HF | Speedup |"
      echo "|---|---:|---:|---:|"
      found=1
    fi
    local run_name
    run_name=$(basename "$file" .log)
    local mini_tps
    mini_tps=$(extract_tok_s_arrow "\\[mini-vLLM\\]" "$file")
    local hf_tps
    hf_tps=$(extract_tok_s_arrow "\\[HF generate\\]" "$file")
    local speedup
    speedup=$(extract_speedup "$file")
    echo "| $run_name | ${mini_tps:-n/a} | ${hf_tps:-n/a} | ${speedup:-n/a} |"
  done
}

print_prefix_cache_table() {
  local file=$1
  [[ -f "$file" ]] || return 0
  print_section "Prefix Cache"
  echo "| Mode | tok/s | Prefix hits | Hit tokens |"
  echo "|---|---:|---:|---:|"
  awk '
    /^=== prefix_cache=/ {
      mode=$0
      sub(/^=== prefix_cache=/, "", mode)
      sub(/ ===$/, "", mode)
    }
    /elapsed=.*tok\/s=/ {
      match($0, /tok\/s=([0-9.]+)/, a)
      tps=a[1]
    }
    /prefix_cache_hits=/ {
      match($0, /prefix_cache_hits=([0-9]+)/, a)
      hits=a[1]
      match($0, /prefix_cache_hit_tokens=([0-9]+)/, b)
      tokens=b[1]
      printf("| %s | %s | %s | %s |\n", mode, tps, hits, tokens)
    }
  ' "$file"
}

print_stress_table() {
  local file=$1
  [[ -f "$file" ]] || return 0
  print_section "Stress"
  echo "| Run | tok/s | avg_ttft_s | avg_itl_s | gpu_blocks_used | swap_out | swap_in |"
  echo "|---|---:|---:|---:|---:|---:|---:|"
  echo "| $(basename "$file" .log) | $(extract_stress_field output_tokens_per_s "$file") | $(extract_stress_field avg_ttft_s "$file") | $(extract_stress_field avg_itl_s "$file") | $(extract_stress_field gpu_kv_blocks_used "$file") | $(extract_stress_field swap_out_count "$file") | $(extract_stress_field swap_in_count "$file") |"
}

print_spec_table() {
  local file=$1
  [[ -f "$file" ]] || return 0
  local base_tps
  base_tps=$(grep -A2 "Baseline" "$file" 2>/dev/null | grep "tok/s" | head -n 1 | sed -E 's/.*\(([0-9.]+ tok\/s)\).*/\1/' | trim)
  local spec_tps
  spec_tps=$(grep -A2 "Speculative decoding" "$file" 2>/dev/null | grep "tok/s" | head -n 1 | sed -E 's/.*\(([0-9.]+ tok\/s)\).*/\1/' | trim)
  local alpha
  alpha=$(extract_first_match "Implied acceptance rate" "$file" | sed -E 's/.*≈ ([0-9.]+).*/\1/' | trim)
  local latency_speedup
  latency_speedup=$(extract_first_match "Latency speedup" "$file" | sed -E 's/.*: ([0-9.]+x).*/\1/' | trim)
  print_section "Speculative"
  echo "| Run | Baseline | Speculative | Acceptance α | Latency speedup |"
  echo "|---|---:|---:|---:|---:|"
  echo "| $(basename "$file" .log) | ${base_tps:-n/a} | ${spec_tps:-n/a} | ${alpha:-n/a} | ${latency_speedup:-n/a} |"
}

print_profile_table() {
  local dir=$1
  local nsys_count=0
  local ncu_count=0
  if find "$dir/profile" -maxdepth 1 \( -name '*.qdrep' -o -name '*.nsys-rep' \) | grep -q . 2>/dev/null; then
    nsys_count=$(find "$dir/profile" -maxdepth 1 \( -name '*.qdrep' -o -name '*.nsys-rep' \) | wc -l | awk '{print $1}')
  fi
  if find "$dir/profile" -maxdepth 1 -name '*.ncu-rep' | grep -q . 2>/dev/null; then
    ncu_count=$(find "$dir/profile" -maxdepth 1 -name '*.ncu-rep' | wc -l | awk '{print $1}')
  fi
  if [[ $nsys_count -gt 0 || $ncu_count -gt 0 ]]; then
    print_section "Profiles"
    echo "| Type | Count |"
    echo "|---|---:|"
    echo "| Nsight Systems reports | $nsys_count |"
    echo "| Nsight Compute reports | $ncu_count |"
  fi
}

echo "# Summary"
echo
echo "- Run directory: \`$RUN_DIR\`"
echo "- Generated at: \`$(date '+%Y-%m-%d %H:%M:%S')\`"

print_kernel_table "$RUN_DIR"
print_throughput_group "Ablations" "$RUN_DIR"/ablations/*.log
print_prefix_cache_table "$RUN_DIR/ablations/prefix_cache.log"
print_throughput_group "Scaling" "$RUN_DIR"/scaling/*.log
print_stress_table "$RUN_DIR/stress/base.log"
print_spec_table "$RUN_DIR/spec/gamma4.log"
print_profile_table "$RUN_DIR"
