import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

voices = ["amy\n(teacher)", "ex02\nstage 1", "ex02\nstage 2", "libritts_r\nspk 3922"]
p50 = [180, 330, 140, 45]
p95 = [260, 443, 205, 80]
mx  = [260, 500, 300, 80]

x = np.arange(len(voices))
w = 0.24
fig, ax = plt.subplots(figsize=(7.2, 4.2))
b1 = ax.bar(x - w, p50, w, label="p50", color="#4C78A8")
b2 = ax.bar(x,     p95, w, label="p95", color="#F58518")
b3 = ax.bar(x + w, mx,  w, label="max", color="#9E9E9E")
ax.set_xticks(x)
ax.set_xticklabels(voices)
ax.set_ylabel("pause duration (ms)")
ax.set_title("Comma pause distributions across voices\n(40 targeted sentences, 3 renders each, noise_scale=0.3, noise_w=0.5)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
for bars in (b1, b2, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 8, f"{int(b.get_height())}",
                ha="center", va="bottom", fontsize=8, color="#333333")
fig.tight_layout()
fig.savefig("fig_comma_pauses.png", dpi=200)
print("figure written")
