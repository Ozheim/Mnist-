import struct
import numpy as np
import os

def parse_labels(filename):
   """Parse MNIST label file (.idx1-ubyte)"""
   with open(filename, 'rb') as f:
       magic, count = struct.unpack('>II', f.read(8))
       
       if magic != 2049:
           raise ValueError(f"Wrong magic number: {magic}, expected 2049")
       
       labels = np.frombuffer(f.read(), dtype=np.uint8)
       return labels

def parse_images(filename):
   """Parse MNIST image file (.idx3-ubyte)"""
   with open(filename, 'rb') as f:
       magic, count, height, width = struct.unpack('>IIII', f.read(16))
       
       if magic != 2051:
           raise ValueError(f"Wrong magic number: {magic}, expected 2051")
       
       images = np.frombuffer(f.read(), dtype=np.uint8)
       images = images.reshape(count, height, width)
       
       return images

def load_mnist_data(data_dir):
   """Load MNIST data from data/ folder"""
   train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
   train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
   test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
   test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
   
   # Load test data
   test_labels = parse_labels(test_labels_path) if os.path.exists(test_labels_path) else None
   test_images = parse_images(test_images_path) if os.path.exists(test_images_path) else None
   
   # Load train data
   train_labels = parse_labels(train_labels_path) if os.path.exists(train_labels_path) else None
   train_images = parse_images(train_images_path) if os.path.exists(train_images_path) else None
   
   return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
   data_dir = "../data"

   try:
       train_data, test_data = load_mnist_data(data_dir)
       print(f"Train images: {train_data[0].shape if train_data[0] is not None else 'None'}")
       print(f"Train labels: {train_data[1].shape if train_data[1] is not None else 'None'}")
       print(f"Test images: {test_data[0].shape if test_data[0] is not None else 'None'}")
       print(f"Test labels: {test_data[1].shape if test_data[1] is not None else 'None'}")
   except Exception as e:
       print(f"Error: {e}")