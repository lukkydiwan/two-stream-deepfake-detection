import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import visualkeras
import pydot

# 1. THE ARCHITECTURE
def build_two_stream_model(input_shape=(224, 224, 3)):
    # 1. Create a generic base model once to download weights correctly
    base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    # --- STREAM 1: RGB ---
    input_rgb = layers.Input(shape=input_shape, name="rgb_input")
    # We use the base_model but give this 'call' a unique name
    x1 = base_model(input_rgb)
    x1 = layers.GlobalAveragePooling2D(name="rgb_avg_pool")(x1)

    # --- STREAM 2: SRM/Noise ---
    input_noise = layers.Input(shape=input_shape, name="noise_input")
    # We use the same base_model here
    x2 = base_model(input_noise)
    x2 = layers.GlobalAveragePooling2D(name="srm_avg_pool")(x2)

    # --- FUSION ---
    combined = layers.Concatenate(name="fusion_layer")([x1, x2])
    x = layers.Dense(256, activation='relu', name="dense_feat")(combined)
    x = layers.Dropout(0.4, name="dropout_layer")(x)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(x)

    model = models.Model(inputs=[input_rgb, input_noise], outputs=output)
    return model

# 2. THE DATA LOADER
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def dual_generator(dir_rgb, dir_srm, batch_size=32, subset='training'):
    gen_rgb = train_datagen.flow_from_directory(
        dir_rgb, target_size=(224, 224), batch_size=batch_size, 
        class_mode='binary', subset=subset, seed=42)
    
    gen_srm = train_datagen.flow_from_directory(
        dir_srm, target_size=(224, 224), batch_size=batch_size, 
        class_mode='binary', subset=subset, seed=42)
    
    while True:
        X1, y1 = next(gen_rgb)
        X2, y2 = next(gen_srm)
        yield [X1, X2], y1
# 3. INITIALIZATION
model = build_two_stream_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])


# 4. VISUALIZATION (UTF-8 Fix)
print("Generating model architecture table...")

# Use encoding='utf-8' to prevent the UnicodeEncodeError
with open('model_architecture_summary.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("- Saved model_architecture_summary.txt")

# 5. THE CRITICAL SANITY CHECK (This is your real Day 2 Goal)
print("\nRunning Sanity Check (1 Step)...")
try:
    # Use the exact folder name from your archive
    train_gen = dual_generator('../archive/train', '../srm_dataset/train', batch_size=4)
    X_batch, y_batch = next(train_gen)
    
    # Check the shapes to be 100% sure
    print(f"RGB Batch Shape: {X_batch[0].shape}") # Should be (4, 224, 224, 3)
    print(f"SRM Batch Shape: {X_batch[1].shape}") # Should be (4, 224, 224, 3)
    
    model.train_on_batch(X_batch, y_batch)
except Exception as e:
    print(f">>> ERROR in Data Loader: {e}")