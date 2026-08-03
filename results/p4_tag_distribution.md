# MTG-Jamendo 标签分布

- 范围：全量元数据（55,609 首，不依赖本地音频）
- 命令：`python -m scripts.jamendo_stats --all`

## 总览

| 项 | 值 |
|---|---|
| 曲目数 | 55,609 |
| 总时长 | 3770 小时 |
| 标签数 | 195 |
| 每首平均标签数 | 4.19 |
| 标签矩阵密度 | 2.15%（**其余 97.9% 是 0**）|
| 最高频标签正样本 | 16,480 |
| 中位标签正样本 | 509 |
| 最低频标签正样本 | 119 |
| 头部 10% 标签占正样本比例 | **48%** |
| 正样本 < 50 的标签数 | **0 / 195** |

## 各类别

| 类别 | 标签数 | 最少 | 中位 | 最多 |
|---|---|---|---|---|
| genre | 95 | 121 | 547 | 16,480 |
| instrument | 41 | 157 | 721 | 7,750 |
| mood/theme | 59 | 119 | 457 | 1,657 |

## 最高频 15 个标签

| 标签 | 正样本 | 占比 |
|---|---|---|
| `genre---electronic` | 16,480 | 29.6% |
| `genre---soundtrack` | 8,094 | 14.6% |
| `genre---pop` | 7,805 | 14.0% |
| `instrument---piano` | 7,750 | 13.9% |
| `genre---ambient` | 7,570 | 13.6% |
| `instrument---synthesizer` | 7,496 | 13.5% |
| `genre---rock` | 6,865 | 12.3% |
| `instrument---drums` | 6,120 | 11.0% |
| `instrument---bass` | 5,726 | 10.3% |
| `genre---classical` | 5,602 | 10.1% |
| `genre---easylistening` | 4,833 | 8.7% |
| `instrument---guitar` | 4,804 | 8.6% |
| `instrument---electricguitar` | 4,241 | 7.6% |
| `genre---experimental` | 3,941 | 7.1% |
| `genre---alternative` | 3,761 | 6.8% |

## 最低频 15 个标签

| 标签 | 正样本 |
|---|---|
| `mood/theme---travel` | 171 |
| `genre---swing` | 169 |
| `genre---choir` | 167 |
| `mood/theme---horror` | 158 |
| `instrument---organ` | 157 |
| `mood/theme---heavy` | 156 |
| `mood/theme---mellow` | 154 |
| `genre---bossanova` | 148 |
| `genre---ethnicrock` | 140 |
| `genre---bluesrock` | 139 |
| `genre---medieval` | 135 |
| `genre---oriental` | 134 |
| `mood/theme---sexy` | 122 |
| `genre---african` | 121 |
| `mood/theme---fast` | 119 |

![标签分布](p4_tag_distribution.png)
