import matplotlib.pyplot as plt
import numpy as np

# var = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
# acc = np.array([0.90, 0.87, 0.81, 0.7, 0.57])

# fig, ax = plt.subplots()
# ax.plot(var, acc, linewidth=2.0)

# fig.suptitle('Perturbation Test', fontsize=20)
# plt.xlabel('Gaussian Noise Variance', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.grid()

# plt.savefig('noisy.png')


rot  = np.array([15, 30, 45, 60, 90])
acc = np.array([0.88, 0.65, 0.35, 0.2, 0.27])

fig, ax = plt.subplots()
ax.plot(rot, acc, linewidth=2.0)

fig.suptitle('Rotation Test', fontsize=20)
plt.xlabel('Degrees', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.grid()

plt.savefig('rot.png')