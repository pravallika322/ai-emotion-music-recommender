import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# 1. BASIC INFO
# ==========================
print("✅ TensorFlow version:", tf.__version__)

# ==========================
# 2. DATASET PATH
# ==========================
DATASET_DIR = r"C:\Users\nidad\OneDrive\Desktop\emotion_music\Dataset"

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset path not found: {DATASET_DIR}")

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# ==========================
# 3. DATA GENERATORS
# ==========================

# Train + Validation (80% train, 20% validation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# Test generator (NO augmentation)
test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

# ==========================
# 4. CREATE GENERATORS
# ==========================

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================
# 5. SANITY CHECKS
# ==========================

print("\n✅ Class Indices (Label Mapping):")
print(train_generator.class_indices)

print("\n✅ Sample Counts:")
print("Train:", train_generator.samples)
print("Validation:", val_generator.samples)
print("Test:", test_generator.samples)

x_batch, y_batch = next(train_generator)
print("\n✅ One Batch Shape:")
print("Images:", x_batch.shape)   # (64, 48, 48, 1)
print("Labels:", y_batch.shape)   # (64, 4)

print("\n🎉 Preprocessing completed successfully!")
