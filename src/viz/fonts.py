"""matplotlib 的中文字体选择。

单独抽出来是因为**每个画图脚本都需要它**，而复制一份的代价不是重复代码，
是"某个脚本忘了调用"——那种失败很隐蔽：stderr 刷一堆 findfont 警告，
图**照样生成**，只是所有中文变成方框，而警告很容易被别的输出淹没。
"""

from __future__ import annotations

# 按优先级排：前面是 macOS 自带的，后面覆盖 Linux / Windows
CANDIDATES = ("PingFang SC", "Heiti SC", "Hiragino Sans GB",
              "Arial Unicode MS", "Noto Sans CJK SC", "Source Han Sans SC",
              "Microsoft YaHei", "SimHei")


def use_cjk_font(strict: bool = False) -> str | None:
    """挑一个可用的中文字体设进 rcParams。

    Args:
        strict: 找不到时抛错而不是打印警告。生成**要提交的图**时应当传 True ——
            一张中文全是方框的图混进仓库，比脚本报错难发现得多。

    Returns:
        选中的字体名；没找到则 ``None``。
    """
    import matplotlib
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATES:
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            # 选了中文字体后负号会变成方块，必须一并关掉 Unicode 减号
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    msg = ("没找到中文字体，图中的中文会显示为方框。"
           f"尝试过：{', '.join(CANDIDATES)}")
    if strict:
        raise RuntimeError(msg)
    print(f"⚠️  {msg}")
    return None
