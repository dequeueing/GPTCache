import matplotlib.pyplot as plt


eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/eps/"
png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/png/"

# Sample data for the two lines
x = [0.2 ,0.4, 0.6, 0.8, 0.9, 0.95, 1.0]
y1 = [0.89, 0.89, 0.87, 0.88, 0.83, 0.77, 0]
y2 = [0.88, 0.87, 0.84, 0.85, 0.8, 0.7, 0]

plt.style.use('ggplot')

# Plotting two lines
plt.plot(x, y1, label='whitebox', color='#457B9D', linewidth=3, marker='o', markersize=7)
plt.plot(x, y2, label='blackbox', color='#E59866', linewidth=3, marker='s', markersize=7)

# range of y
plt.ylim(0, 1)

# Add labels with bold and large font
plt.xlabel('Similarity threshold', fontsize=16, fontweight='bold')
plt.ylabel('ASR', fontsize=16, fontweight='bold')

# Set specific x-ticks and bold font for ticks
plt.xticks([0.2 ,0.4, 0.6, 0.8, 0.9, 0.95, 1.0], fontsize=14, fontweight='bold', rotation=45)
plt.yticks(fontsize=14, fontweight='bold')

# Legend with bold and large font
plt.legend(fontsize=14, frameon=True)

# Adjust layout and save
eps_file = eps_dir + 'res:threshold.eps'
png_file = png_dir + 'res:threshold.png'
plt.tight_layout()
plt.savefig(eps_file, format='eps')
plt.savefig(png_file, format='png')
plt.show()
