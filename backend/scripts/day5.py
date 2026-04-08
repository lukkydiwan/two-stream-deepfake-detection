import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
# --- 1. DATA AUGMENTATION SETUP (The Day 5 Secret Sauce) ---
# This forces the model to stop memorizing and start generalizing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,      # Head tilts
    width_shift_range=0.1,  # Horizontal shifts
    height_shift_range=0.1, # Vertical shifts
    shear_range=0.1,        # Geometric distortion
    zoom_range=0.1,         # Zooming into skin texture
    horizontal_flip=True,   # Mirrored faces
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# --- 2. DUAL GENERATOR LOGIC ---
def dual_generator(dir_rgb, dir_srm, batch_size=32, subset='training', datagen=train_datagen):
    gen_rgb = datagen.flow_from_directory(
        dir_rgb, target_size=(224, 224), batch_size=batch_size, 
        class_mode='binary', subset=subset, seed=42)
    
    gen_srm = datagen.flow_from_directory(
        dir_srm, target_size=(224, 224), batch_size=batch_size, 
        class_mode='binary', subset=subset, seed=42)
    
    while True:
        X1, y1 = next(gen_rgb)
        X2, y2 = next(gen_srm)
        # Dictionary mapping for multi-input stability
        yield ({"rgb_input": X1, "noise_input": X2}, y1)

# --- 3. PATHS & GENERATOR INITIALIZATION ---
rgb_path = '../archive/train'
srm_path = '../srm_dataset/train'

train_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='training', datagen=train_datagen)
val_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='validation', datagen=val_datagen)

# --- 4. LOAD & RE-COMPILE THE MODEL ---
print("Loading the fine-tuned model from Day 4...")
# We load the best model you just saved

model = load_model('final_tuned_model.keras', compile=False)

# We use a very low learning rate to "polish" the weights without breaking them
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# --- 5. CALLBACKS ---
callbacks = [
    ModelCheckpoint('final_robust_model.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    CSVLogger('day5_augmented_logs.csv', append=False),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
]

# --- 6. START THE FINAL TRAINING BURST ---
print("\nStarting Day 5: Robust Augmented Training...")
# Increased steps_per_epoch because augmentation creates "new" data
history = model.fit(
    train_gen,
    steps_per_epoch=15, 
    validation_data=val_gen,
    validation_steps=4,
    epochs=15,
    callbacks=callbacks
)

print("\nTraining Complete! You now have 'final_robust_model.keras' and 'day5_augmented_logs.csv'.")