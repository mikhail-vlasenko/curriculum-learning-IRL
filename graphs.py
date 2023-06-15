import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('graph_data/fixed_airl.csv')
old_runs = [121, 122, 123]
new_runs = [162, 163, 166]
columns = []
for c in df.columns:
    if "__MIN" not in c and "__MAX" not in c:
        columns.append(c)
df = df[columns]
df.fillna(method='ffill', inplace=True)

old_columns = []
new_columns = []
for c in df.columns:
    for o in old_runs:
        if str(o) in c:
            old_columns.append(c)
    for n in new_runs:
        if str(n) in c:
            new_columns.append(c)

df['old_mean'] = df[old_columns].mean(axis=1)
df['new_mean'] = df[new_columns].mean(axis=1)
df['old_std'] = df[old_columns].std(axis=1)
df['new_std'] = df[new_columns].std(axis=1)

fig, ax = plt.subplots()

ax.plot(df['Step'], df['old_mean'], label='old')
ax.plot(df['Step'], df['new_mean'], label='new')
ax.fill_between(df['Step'], df['old_mean'] - df['old_std'], df['old_mean'] + df['old_std'], alpha=0.4)
ax.fill_between(df['Step'], df['new_mean'] - df['new_std'], df['new_mean'] + df['new_std'], alpha=0.4)
# ax.fill_between(xs, lower, upper, alpha=0.4)

# ax.set_xlabel('$n$')
# ax.set_ylabel('$m$ [%]')
ax.legend()
ax.set_title('True reward in the environment')
plt.show()
