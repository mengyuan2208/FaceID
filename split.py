import random
import os

base_dir = 'aligned_images_DB'
pictures = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            pictures.append(full_path)

random.shuffle(pictures)

split_ratio = 0.8
split_index = int(len(pictures) * split_ratio)

train_pictures = pictures[:split_index]
val_pictures = pictures[split_index:]

with open('train_pictures.txt', 'w') as train_file:
    for picture in train_pictures:
        train_file.write(f"{picture}\n")

with open('val_pictures.txt', 'w') as val_file:
    for picture in val_pictures:
        val_file.write(f"{picture}\n")