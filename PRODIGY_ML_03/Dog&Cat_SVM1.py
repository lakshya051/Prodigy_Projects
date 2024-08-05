import os;
import numpy as np
import cv2
import time
import pickle


start_time = time.time()

dir = 'D:\\projects\\Prodigy\\task_3\\dogs-vs-cats\\train'
file = os.listdir(dir)


train_cat_dir = list()
train_dog_dir = list()

for f in file:
    target = f.split(".")[0]
    full_path = os.path.join(dir,f)

    if (target == "cat"):
        train_cat_dir.append(full_path)

    if (target == "dog"):
        train_dog_dir.append(full_path)

# Define the target size for resizing
IMG_SIZE = (64, 64)

# Lists to store the data and labels
data = []
labels = []

# Function to process images
def process_images(file_list, label):
    for file_path in file_list:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Read the image in color
        if img is not None:
            img_resized = cv2.resize(img, IMG_SIZE)  # Resize the image
            img_normalized = img_resized / 255.0  # Normalize the image
            img_flattened = np.array(img_normalized).flatten()  # Flatten the image
            data.append([img_flattened,label])


# Process cat images
process_images(train_cat_dir, 0)

# Process dog images
process_images(train_dog_dir, 1)

pick_in = open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()

print(f"Saving Time: {time.time() - start_time} seconds")