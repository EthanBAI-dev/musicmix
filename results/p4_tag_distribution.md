# MTG-Jamendo 标签分布

- 范围：本地子样本 · autotagging_top50tags · split-0（5404 首）
- 命令：`python -m scripts.jamendo_stats --subset autotagging_top50tags --min-positives 0`

## 总览

| 项 | 值 |
|---|---|
| 曲目数 | 5,404 |
| 总时长 | 358 小时 |
| 标签数 | 50 |
| 每首平均标签数 | 3.10 |
| 标签矩阵密度 | 6.20%（**其余 93.8% 是 0**）|
| 最高频标签正样本 | 1,596 |
| 中位标签正样本 | 217 |
| 最低频标签正样本 | 120 |
| 头部 10% 标签占正样本比例 | **28%** |
| 正样本 < 50 的标签数 | **0 / 50** |

## 各类别

| 类别 | 标签数 | 最少 | 中位 | 最多 |
|---|---|---|---|---|
| genre | 31 | 120 | 231 | 1,596 |
| instrument | 14 | 123 | 227 | 805 |
| mood/theme | 5 | 141 | 155 | 188 |

## 各 split

| split | 曲目数 | 时长(h) |
|---|---|---|
| train | 3,147 | 214 |
| validation | 1,144 | 71 |
| test | 1,113 | 73 |

## 逐标签阈值调优的可行性（L4）

验证集共 1,144 首。**正样本 < 10 的标签有 0 / 50 个** —— 在这些标签上搜阈值等于过拟合噪声。

最稀薄的 10 个：无

## 最高频 15 个标签

| 标签 | 正样本 | 占比 |
|---|---|---|
| `genre---electronic` | 1,596 | 29.5% |
| `genre---soundtrack` | 805 | 14.9% |
| `instrument---piano` | 805 | 14.9% |
| `genre---pop` | 765 | 14.2% |
| `genre---ambient` | 745 | 13.8% |
| `instrument---synthesizer` | 732 | 13.5% |
| `genre---rock` | 666 | 12.3% |
| `genre---classical` | 582 | 10.8% |
| `instrument---drums` | 574 | 10.6% |
| `genre---easylistening` | 539 | 10.0% |
| `instrument---bass` | 527 | 9.8% |
| `instrument---guitar` | 522 | 9.7% |
| `instrument---electricguitar` | 398 | 7.4% |
| `genre---chillout` | 375 | 6.9% |
| `genre---experimental` | 347 | 6.4% |

## 最低频 15 个标签

| 标签 | 正样本 |
|---|---|
| `instrument---voice` | 167 |
| `instrument---strings` | 161 |
| `genre---funk` | 157 |
| `genre---atmospheric` | 156 |
| `mood/theme---relaxing` | 155 |
| `genre---instrumentalpop` | 153 |
| `instrument---drummachine` | 144 |
| `mood/theme---emotional` | 142 |
| `mood/theme---energetic` | 141 |
| `genre---trance` | 138 |
| `genre---downtempo` | 135 |
| `genre---triphop` | 131 |
| `instrument---electricpiano` | 123 |
| `genre---reggae` | 121 |
| `genre---metal` | 120 |

![标签分布](p4_tag_distribution.png)
