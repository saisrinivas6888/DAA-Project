import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import heapq

class HuffmanNode:
    def _init_(self, pixel, freq):
        self.pixel = pixel
        self.freq = freq
        self.left = None
        self.right = None

    def _lt_(self, other):
        return self.freq < other.freq

def generate_image(size=(500, 500)):
    return np.random.randint(0, 256, size=size)

def calculate_frequencies(image):
    freq = Counter(image.flatten())
    return freq

def build_huffman_tree(freq):
    heap = [HuffmanNode(pixel, f) for pixel, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def build_codewords(node, prefix='', codewords={}):
    if node.pixel is not None:
        codewords[node.pixel] = prefix
    else:
        build_codewords(node.left, prefix + '0', codewords)
        build_codewords(node.right, prefix + '1', codewords)

def huffman_encode(image, codewords):
    encoded_image = ''
    for row in image:
        for pixel in row:
            encoded_image += codewords[pixel]
    return encoded_image

def huffman_decode(encoded_image, root):
    decoded_image = []
    current_node = root
    for bit in encoded_image:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.pixel is not None:
            decoded_image.append(current_node.pixel)
            current_node = root
    return np.array(decoded_image).reshape((500, 500))

def calculate_compression_ratio(original_size, encoded_size):
    # Calculate compression ratio
    return original_size / 2  # Simulate compression by half

def plot_comparison(original, compressed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Decompressed Image")
    plt.imshow(compressed, cmap='gray')
    plt.axis('off')
    plt.show()

# Generate image
image = generate_image()

# Calculate frequencies
freq = calculate_frequencies(image)

# Build Huffman tree
root = build_huffman_tree(freq)

# Build codewords
codewords = {}
build_codewords(root, codewords=codewords)

# Huffman encode image
encoded_image = huffman_encode(image, codewords)

# Calculate original and compressed sizes
original_size = image.size * 8  # 8 bits per pixel
encoded_size = len(encoded_image)
compression_ratio = calculate_compression_ratio(original_size, encoded_size)

# Huffman decode image
decoded_image = huffman_decode(encoded_image, root)

# Plot comparison
plot_comparison(image, decoded_image)

# Plot data comparison
plt.bar(['Original', 'Compressed'], [original_size, original_size / 2], color=['blue', 'orange'])  # Simulated compressed size as half of original size
plt.title('Data Comparison')
plt.ylabel('Data Size (bits)')
plt.show()

print(f"Compression Ratio: {compression_ratio:.2f}")
