#!/bin/zsh
# 整夜实验队列。每一步都**可重入**：产物已存在就跳过，
# 所以被杀掉之后重跑这个脚本会从断点接着走。
#
# 顺序是按「便宜且能决定后续」排的：
#   1. 8 段训练      —— 定饱和点（4→8 还涨吗）
#   2. 4 段上重跑消融 —— P4 的四个自研点是在较差输入上判死的，可能翻案
#   3. 330M × 4 段   —— 两个已验证的正向改动是否叠加（最贵，放最后）
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/music-mix/bin/python
LOG=results/overnight.log
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

say "=== 队列启动 ==="

# ---------- 步骤 0：等 8 段特征提完 ----------
say "步骤0 等待 8 段特征提取完成…"
while :; do
  n=$(find data/jamendo/features/MERT-v1-95M_L6_s5_30s_x8 -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 5400 ] && break
  pgrep -f "extract_backbone --segments 8" >/dev/null || { say "  ⚠️ 提取进程没了（$n/5404），重启"; \
    $PY -m scripts.extract_backbone --segments 8 --layer 6 >> /tmp/x8.log 2>&1 & }
  sleep 120
done
say "步骤0 ✅ 8 段特征齐了（$n）"

# ---------- 步骤 1：8 段训练，定饱和点 ----------
for seed in 0 1 2 3 4; do
  out=results/seeds/SEGall8_s${seed}.json
  [ -f "$out" ] && continue
  $PY -m scripts.train_tagging --level L2 --arch attnhead --pooling mean \
      --layer 6 --segments 8 --seed $seed --out "$out" >> /tmp/seg8.log 2>&1 \
    && say "步骤1 SEGall8 seed$seed ✅" || say "步骤1 SEGall8 seed$seed ❌"
done
say "步骤1 ✅ 8 段训练完成"

# ---------- 步骤 2：4 段输入上重跑 P4 消融 ----------
say "步骤2 开始（25 个 run）"
./scripts/run_ablation_4seg.sh >> $LOG 2>&1 && say "步骤2 ✅" || say "步骤2 ❌"

# ---------- 步骤 3：330M × 4 段 ----------
say "步骤3 提 330M 4 段特征（最贵的一步）"
$PY -m scripts.extract_backbone --model m-a-p/MERT-v1-330M --layer 6 --segments 4 \
    >> /tmp/m330x4.log 2>&1 && say "步骤3 提取 ✅" || say "步骤3 提取 ❌"

for seed in 0 1 2 3 4; do
  out=results/seeds/M330SEG4_s${seed}.json
  [ -f "$out" ] && continue
  $PY -m scripts.train_tagging --level L2 --arch attnhead --pooling mean \
      --backbone m-a-p/MERT-v1-330M --layer 6 --segments 4 \
      --seed $seed --out "$out" >> /tmp/m330seg.log 2>&1 \
    && say "步骤3 M330SEG4 seed$seed ✅" || say "步骤3 M330SEG4 seed$seed ❌"
done

say "=== 队列全部完成 ==="
