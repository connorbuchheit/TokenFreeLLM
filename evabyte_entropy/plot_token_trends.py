import numpy as np
import matplotlib.pyplot as plt


num_bytes = np.array([152, 269, 349, 586, 843, 1292, 1695, 3098, 5899, 8005, 11843])

tiktoken_total = np.array([23, 39, 52, 90, 126, 199, 261, 488, 960, 1324, 1959])
tiktoken_unique = np.array([22, 32, 42, 68, 94, 136, 173, 284, 480, 549, 716])
tiktoken_avg_length = np.array([6.61, 6.90, 6.71, 6.51, 6.69, 6.49, 6.49, 6.35, 6.14, 6.05, 6.05])

entropy_total = np.array([45, 73, 95, 139, 216, 321, 452, 774, 1338, 1823, 2801])
entropy_unique = np.array([35, 53, 61, 94, 127, 172, 222, 367, 613, 740, 1025])
entropy_avg_length = np.array([3.38, 3.68, 3.67, 4.22, 3.90, 4.02, 3.75, 4.00, 4.41, 4.39, 4.23])

derivative_total = np.array([24, 35, 50, 69, 105, 152, 211, 386, 699, 994, 1464])
derivative_unique = np.array([23, 33, 43, 64, 92, 129, 171, 295, 477, 616, 846])
derivative_avg_length = np.array([6.09, 7.66, 6.96, 8.48, 8.02, 8.49, 8.03, 8.02, 8.44, 8.05, 8.09])


# Create a figure with 3 subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Total tokens
axs[0].plot(num_bytes, tiktoken_total, label='Tiktoken')
axs[0].plot(num_bytes, entropy_total, label='Entropy')
axs[0].plot(num_bytes, derivative_total, label='Derivative')
axs[0].set_title('Total Tokens')
axs[0].set_xlabel('Number of Bytes')
axs[0].set_ylabel('Number of Tokens')
axs[0].grid(True)
axs[0].legend()

# Plot 2: Unique tokens
axs[1].plot(num_bytes, tiktoken_unique, label='Tiktoken')
axs[1].plot(num_bytes, entropy_unique, label='Entropy')
axs[1].plot(num_bytes, derivative_unique, label='Derivative')
axs[1].set_title('Unique Tokens')
axs[1].set_xlabel('Number of Bytes')
axs[1].set_ylabel('Number of Unique Tokens')
axs[1].grid(True)
axs[1].legend()

# Plot 3: Average token length
axs[2].plot(num_bytes, tiktoken_avg_length, label='Tiktoken')
axs[2].plot(num_bytes, entropy_avg_length, label='Entropy')
axs[2].plot(num_bytes, derivative_avg_length, label='Derivative')
axs[2].set_title('Average Token Length')
axs[2].set_xlabel('Number of Bytes')
axs[2].set_ylabel('Average Length (bytes)')
axs[2].grid(True)
axs[2].legend()


plt.tight_layout()
plt.savefig('token_trends_combined.png')
plt.show()  

