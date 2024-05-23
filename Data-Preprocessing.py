# src/data_preprocessing.py

import os
import pandas as pd
import numpy as np
import cv2
import json
from sklearn.preprocessing import StandardScaler

# Define file paths
raw_data_path = 'data/raw/'
processed_data_path = 'data/processed/'

# Create directories if they don't exist
os.makedirs(processed_data_path, exist_ok=True)

# Function to load and preprocess camera images
def preprocess_images(image_dir):
    processed_images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            # Resize image
            image = cv2.resize(image, (224, 224))
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_images.append(image)
    return np.array(processed_images)

# Function to load and preprocess LiDAR data
def preprocess_lidar(lidar_file):
    # Assuming LiDAR data is in a CSV format
    lidar_data = pd.read_csv(lidar_file)
    return lidar_data.values

# Function to load and preprocess GPS data
def preprocess_gps(gps_file):
    gps_data = pd.read_csv(gps_file)
    return gps_data.values

# Function to load and preprocess IMU data
def preprocess_imu(imu_file):
    imu_data = pd.read_csv(imu_file)
    return imu_data.values

# Load and preprocess all data
print("Loading and preprocessing data...")

# Load camera images
image_dir = os.path.join(raw_data_path, 'images')
images = preprocess_images(image_dir)
np.save(os.path.join(processed_data_path, 'images.npy'), images)

# Load LiDAR data
lidar_file = os.path.join(raw_data_path, 'lidar.csv')
lidar_data = preprocess_lidar(lidar_file)
np.save(os.path.join(processed_data_path, 'lidar.npy'), lidar_data)

# Load GPS data
gps_file = os.path.join(raw_data_path, 'gps.csv')
gps_data = preprocess_gps(gps_file)
np.save(os.path.join(processed_data_path, 'gps.npy'), gps_data)

# Load IMU data
imu_file = os.path.join(raw_data_path, 'imu.csv')
imu_data = preprocess_imu(imu_file)
np.save(os.path.join(processed_data_path, 'imu.npy'), imu_data)

# Standardize GPS and IMU data
print("Standardizing GPS and IMU data...")

scaler = StandardScaler()
gps_data = scaler.fit_transform(gps_data)
imu_data = scaler.fit_transform(imu_data)

np.save(os.path.join(processed_data_path, 'gps_standardized.npy'), gps_data)
np.save(os.path.join(processed_data_path, 'imu_standardized.npy'), imu_data)

print("Data preprocessing completed! Processed data saved to 'data/processed/'")
