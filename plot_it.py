import matplotlib.pyplot as plt

# Data for OpenPack (left wrist)
left_wrist_percentages = ['25%', '50%', '75%', '100%']
left_wrist_config_names = ['no pretrain', 'pretrain how2sign multitask', 'pretrain grab multitask', 'pretrain both multitask', 'openpack-dcl-leftwrist-wo virIMU', 'openpack-dcl-leftwrist-w virIMU']
left_wrist_performance_metrics = {
    'no pretrain': [23.11, 30.75, 25.19, 31.43],
    'pretrain how2sign multitask': [29.82, 38.15, 33.09, 42.32],
    'pretrain grab multitask': [28.36, 32.18, 30.07, 34.08],
    'pretrain both multitask': [30.33, 37.77, 33.5, 39.83],
    'openpack-dcl-leftwrist-wo virIMU': [28.22, 36.09, 25.27, 38.94],
    'openpack-dcl-leftwrist-w virIMU': [29.00, 29.74, 18.21, 32.39]
}

# Data for OpenPack (both wrists)
both_wrists_percentages = ['25%', '50%', '75%', '100%']
both_wrists_config_names = ['no pretrain', 'pretrain how2sign multitask', 'pretrain grab multitask', 'pretrain both multitask', 'openpack-dcl-bothwrist-wo virIMU', 'openpack-dcl-bothwrist-w virIMU']
both_wrists_performance_metrics = {
    'no pretrain': [33.17, 46.95, 40.06, 53.84],
    'pretrain how2sign multitask': [37.53, 53.67, 45.6, 61.74],
    'pretrain grab multitask': [35.35, 48.53, 41.94, 55.12],
    'pretrain both multitask': [37.11, 51.19, 44.15, 60.23],
    'openpack-dcl-bothwrist-wo virIMU': [26.25, 41.43, 34.15, 49.68],
    'openpack-dcl-bothwrist-w virIMU': [43.59, 29.74, 28.07, 38.82]
}

# Plotting both subgraphs
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 20))

# Plot for OpenPack (left wrist)
ax1 = axes[0]
for method, values in left_wrist_performance_metrics.items():
    ax1.plot(left_wrist_percentages, values, marker='o', label=method)

ax1.set_title('OpenPack (left wrist)')
ax1.set_ylabel('F1 Score')
ax1.legend(loc='lower right')
ax1.grid(True)

# Plot for OpenPack (both wrists)
ax2 = axes[1]
for method, values in both_wrists_performance_metrics.items():
    ax2.plot(both_wrists_percentages, values, marker='o', label=method)

ax2.set_title('OpenPack (both wrists)')
ax2.set_ylabel('F1 Score')
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show()
