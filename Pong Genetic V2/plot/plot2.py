import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

mean_scores = []
all_scores = []
std_scores = []

for filename in sorted(glob.glob('best_paddle_scores/mean_scores*.txt'), key=lambda x:float(re.findall("(\d+)",x)[0])):
    mean_scores.append(np.loadtxt(filename))

for filename in sorted(glob.glob('best_paddle_scores/scores*.txt'), key=lambda x:float(re.findall("(\d+)",x)[0])):
    all_scores.append(np.loadtxt(filename))

mean_scores = np.array(mean_scores)
all_scores = np.array(all_scores)
for scores in all_scores:
    std_scores.append(np.std(scores))


# Plot 
colors = ["tab:blue", "tab:cyan"]


for i in range(all_scores.shape[0]):
    c = colors[i % len(colors)]
    plt.plot([i+1 for _ in range(all_scores.shape[1])], all_scores[i], ".", color=c, alpha=0.2)

# mean
plt.plot(np.arange(1, 51), mean_scores, color="mediumblue", linewidth=1.0, label="mean")
# mean + std
plt.plot(np.arange(1, 51), mean_scores+std_scores, color="#111111", linestyle="-", linewidth=1, label="mean+-std")
# mean - std
plt.plot(np.arange(1, 51), mean_scores-std_scores, color="#111111", linestyle="-", linewidth=1)

plt.xlabel("generations", fontsize=16)
plt.ylabel("score", fontsize=16)
plt.legend(fontsize=12)
plt.show()