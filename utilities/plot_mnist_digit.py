import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
plt.imshow(digits.images[0], cmap='gray')
plt.axis('off')
plt.savefig('digit.png', bbox_inches='tight', pad_inches=0)
plt.close()
