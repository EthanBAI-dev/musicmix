# 云端训练：先下载全量数据，再微调音乐标签模型

新入口是 `train_mert.ipynb`，不是旧 `extract_features.ipynb`。
本次训练**音乐标签模型**，输出一首歌的风格、情绪、乐器概率；它不会让 Demucs 的音轨分离自动变好。
`distill.ipynb` 是另一项分离蒸馏实验，不能与这里的标签微调混为一谈。

## 1. 把项目和数据留在你的云盘

默认目录沿用 `/content/drive/MyDrive/Audio AI/MusicMixer`；如你的真实文件夹不同，改 notebook 的 BASE。

```text
MusicMixer/
  project/                  本次更新后的项目源代码
  data/jamendo/
    meta/splits/split-0/     完整官方训练/验证/测试划分
    audio_tar/              00–99 原始音频 tar + 校验收据
  huggingface/              基座缓存
  MERT-v1-95M-revision.json  固定模型版本
  runs/<实验名>/            checkpoint、预测、配置、日志
```

把本地 `colab/train_mert.ipynb` 上传到 Colab。项目尚未推送时，不要只 clone 旧 GitHub 版本：将交付的 `musicmix-cloud-source.zip` 解压到 Drive 的 project，使 `project/scripts/cloud_train.py` 存在；notebook 会检查新训练脚本是否存在。压缩包不含数据、权重或环境，只含本次训练需要的代码、文档和测试。
已有 tar 可复用，但须采用上面的目录和文件名 `raw_30s_audio-00.tar`，首次执行下载器会校验它们，不要重复创建两份全量数据。

1. CPU 运行时先跑挂载/依赖，再设 `DOWNLOAD_ALL=True`。下载 0–99，而不是从 10 开始。
2. 下载到 Drive，**不解压全量音频，不删除你的源数据**。全量约 500–530 GB，实际以文件大小为准，建议给数据留至少 600 GB，另留 checkpoint 空间。
3. 用 `.partial` 续传；校验每个预期曲目是否在 tar 中、成员边界和 SHA256。中断后同一命令重跑。`--verify-existing` 可重新做全文件校验。
4. SHA 是首次下载后的本地完整性记录，不等于官方签名。若缺失成员或 checksum 改变会停止，保留文件供排查，不自动删除或跳过坏样本。

Drive 5 TB ≠ 系统 RAM ≠ GPU 显存 ≠ `/content` 临时盘。Colab 资源会变化；付费也不保证固定 GPU 或无限时长。[官方说明](https://research.google.com/colaboratory/faq.html)

## 2. 真正训练什么

```text
随机音频片段 → 24kHz 单声道 → 有效样本归一化 → MERT
                                              ↓
                                    掩码平均池化 → MLP → 50 个 logits
                                              ↓
                                    BCE 损失 → 反向传播更新参数
```

- `frozen`：MERT 不更新，只训练标准 MLP 分类头。建立同口径对照。
- `last`：更新最后 `--last-n 2` 个 Transformer 层及分类头，其他参数冻结。
- `full`：更新全部基座和分类头；显存和训练风险更大，前两项跑通后再试。

不是从零预训练一个音乐大模型，也不是新发明的架构，是**成熟预训练模型的任务微调**。
默认最后一层输出，和旧“第6/7层冻结缓存”不是同一实验设置。新旧结果不能直接归因为微调收益，必须用新管线的 frozen 作为对照。

默认 95M、10秒、micro-batch=2、梯度累积8，有效 batch通常16首；最后不足一组按真实样本数缩放。
固定学习率：基座 `1e-5`，分类头 `1e-3`；AdamW；BCE；CUDA FP16 + GradScaler + 梯度裁剪。
没有默认开启 LoRA、梯度检查点、复杂损失、全曲注意力。先保持变量少且明确。95M 参数少不代表一定不会 OOM。

## 3. 按顺序执行

1. 切换 GPU 运行时，再跑训练配置。首次自动解析官方 checkpoint 的 commit SHA，保存在 Drive，以后不随 main 漂移。`trust_remote_code=True` 会执行该仓库代码，应审阅固定版本。
2. 保持 `SMOKE=True`，64首/每个划分，一轮检查音频解码、前向、反向和写盘。此结果**不能报告为正式性能**。
3. 成功后设 `SMOKE=False`，新实验名，先 `MODE='frozen'`，再相同配置 `MODE='last'`。脚本不会自动把旧权重当新实验续训。
4. 断线后重跑挂载/配置，保留完全相同实验配置，设 `RESUME=True`。训练过程中每200个优化器步保存；可改 `--save-every 50` 降低中断损失，但 Drive 写盘更频繁。
5. 比较验证 mAP，之后做 seed 0/1/2；同一曲目集、标签集、时长、验证分段数、训练预算。最后才设 `FINAL_TEST=True`。

## 4. 怎么判断多久、是否卡住

不承诺“40分钟跑完”。耗时取决于分配的 GPU、音频解码、Drive 带宽和训练模式。
完整一轮会遍历全部训练曲目一次；训练随机取一段，验证均匀取三段并平均概率，因此下载整首不等于每轮听完整首。

跑完冒烟后读 `history.jsonl`。正式训练的粗估：
`每轮时间 ≈ 训练曲目数 × 实测每首训练耗时 + 验证曲目数 × 3 × 实测每段推理耗时 + I/O与保存`。
冒烟受首次模型加载/编译影响，正式第一轮更有参考价值。日志每20个优化器步显示进度。

当前先以 chunk 为单位打乱，再在 chunk 内打乱曲目，减少 Drive 随机跨文件访问。每首训练曲目每轮出现一次，但不是全局随机排列；所有对照必须使用相同采样器。
tar 直接读取避免解压数万文件，但远程随机 I/O 仍可能拖慢 GPU。GPU 利用率低且解码等待长时，下一步应做**有容量上限的本地分片暂存**，不是增大模型。

## 5. 输出与恢复边界

- `best.pt`：验证 mAP 最好的完整模型、验证阈值、标签顺序、配置及训练状态。
- `last.pt`：模型、AdamW、AMP scaler、Python/NumPy/torch/CUDA随机状态、epoch和已完成batch游标；恢复到最近一次完整保存的优化器边界。
- `run.json`：基座版本、实际曲目列表、标签和配置签名、Git HEAD、设备；`environment.txt` 保存包版本。
- `validation_predictions.npz` / `test_predictions.npz`：曲目ID、真实标签、预测概率；可用于分风格失败分析和配对统计。
- 测试只有明确的独立命令才运行；阈值来自 best checkpoint 的验证集，不从测试调参。

同目录只允许一个训练进程，不能多个 Colab 会话同时写。`.partial` + rename 避免通常的半写文件，但 Drive 挂载不等于事务型数据库；保留 best/last 两份，不加载来源不明的 `.pt`（pickle有执行风险）。
恢复配置签名校验元数据和参数，**尚不逐次重算全部音频内容的哈希**。数据完整性依赖下载校验收据；修改原始音频后应建新数据版本。
GPU与依赖变化仍可能造成数值差异，不承诺跨设备逐位一致。

## 6. 验证状态与许可

本地相关测试 **55项通过**，覆盖三种梯度范围、tar读音频、真实优化循环、模拟中断后的逐位权重一致性、显式测试和 notebook 语法。
另外，在隔离的 Transformers 4.44.2 环境，官方 MERT-95M/330M 模型代码配缩小随机权重的 **2项兼容性测试通过**，各自覆盖三种模式的前向/反向。
这些不冒充完整预训练 MERT 在 Colab CUDA 上的验证。**完整权重加载、显存和正式训练指标仍须首次 Colab 冒烟确认。**
依赖固定到单独标签训练环境，不重装 Colab 的 CUDA torch；不要把旧项目依赖一股脑安装，尤其其 Python `<3.12` 限制可能与当前 Colab 不匹配。

MERT 权重非商业许可；Jamendo 的数据集用途也有限制。微调不会自动消除原权重和数据的限制。见架构审查报告，企业商业使用需另行确认授权。
