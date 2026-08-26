#!/bin/zsh
# P6 步骤 3：在**全曲 4 段**输入上重跑 P4 的全部消融。
#
# 为什么值得重跑：P4 判死的四个自研点（注意力池化、max 池化、Focal、ASL）
# 全部是在「中间 30 秒」这个**已被证明较差**的输入上判的。
# 尤其注意力池化 —— 它的设计初衷正是「在长跨度里挑出重要片段」，
# 而此前它从没拿到过真正的长跨度输入。
#
# 基线 SEGall4（mean 池化 + BCE）已在 P5b 跑完，这里只补其余配置。
set -e
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/music-mix/bin/python
COMMON=(--level L2 --arch attnhead --layer 6 --segments 4)

for seed in 0 1 2 3 4; do
  for spec in \
      "A4attn:--pooling attention:--loss bce" \
      "A4max:--pooling max:--loss bce" \
      "A4focal:--pooling mean:--loss focal" \
      "A4asl:--pooling mean:--loss asl" \
      "A4linear:--arch linear --pooling mean:--loss bce"; do
    name="${spec%%:*}"; rest="${spec#*:}"; a="${rest%%:*}"; b="${rest##*:}"
    out="results/seeds/${name}_s${seed}.json"
    [ -f "$out" ] && continue
    $PY -m scripts.train_tagging "${COMMON[@]}" ${=a} ${=b} \
        --seed $seed --out "$out" >> /tmp/ablation4seg.log 2>&1
    echo "done $name seed$seed"
  done
done
echo "ALL DONE"
