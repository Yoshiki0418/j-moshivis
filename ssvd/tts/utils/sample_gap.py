import numpy as np

def sample_gap_ms():
    """
    日本語会話の応答分布に基づいた
    自然な沈黙・オーバーラップをランダム生成
    """
    # 正の側（沈黙）: 平均200ms, 標準偏差500ms
    pos_gap = np.random.normal(loc=200, scale=500)
    while pos_gap < 0:
        pos_gap = np.random.normal(loc=200, scale=500)

    # 負の側（オーバーラップ）: 平均-100ms, 標準偏差80ms
    neg_gap = np.random.normal(loc=-100, scale=80)

    # サンプリング確率（正:負 = 0.93:0.07 に設定）
    if np.random.rand() < 0.93:
        gap = pos_gap
    else:
        gap = neg_gap

    # 極端な値は除外（-500ms〜2000msの範囲にクリップ）
    gap = np.clip(gap, -500, 2000)

    return int(gap)

# # テスト: サンプルを確認
# samples = [sample_gap_ms() for _ in range(50000)]
# print(samples)

# import matplotlib.pyplot as plt

# # ヒストグラムを描画して保存
# plt.figure(figsize=(8,5))
# plt.hist(samples, bins=20, color="skyblue", edgecolor="black")
# plt.title("サンプル応答間隔（500件）")
# plt.xlabel("Gap (ms)")
# plt.ylabel("Count")
# plt.axvline(0, color="red", linestyle="--", label="0ms (オーバーラップ境界)")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)

# # 保存先ファイル
# out_file = "gap_distribution.png"
# plt.savefig(out_file, dpi=150, bbox_inches="tight")
# print(f"✅ 図を保存しました: {out_file}")