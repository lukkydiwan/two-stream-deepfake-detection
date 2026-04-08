import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

# --- 1. THE ARCHITECTURE ---
def build_two_stream_model(input_shape=(224, 224, 3)):
    base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    input_rgb = layers.Input(shape=input_shape, name="rgb_input")
    x1 = base_model(input_rgb)
    x1 = layers.GlobalAveragePooling2D(name="rgb_avg_pool")(x1)

    input_noise = layers.Input(shape=input_shape, name="noise_input")
    x2 = base_model(input_noise)
    x2 = layers.GlobalAveragePooling2D(name="srm_avg_pool")(x2)

    combined = layers.Concatenate(name="fusion_layer")([x1, x2])
    x = layers.Dense(256, activation='relu', name="dense_feat")(combined)
    x = layers.Dropout(0.4, name="dropout_layer")(x)
    output = layers.Dense(1, activation='sigmoid', name="prediction")(x)

    return models.Model(inputs=[input_rgb, input_noise], outputs=output)

# --- 2. DATA GENERATOR LOGIC ---
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
        # Yield a TUPLE of inputs (X1, X2) and labels y1
        # This matches the expected Keras 3 multi-input signature
        yield ({"rgb_input": X1, "noise_input": X2}, y1)


# --- 3. PATH SETUP ---
# Update these to match your folder names exactly
rgb_path = '../archive/train' 
srm_path = '../srm_dataset/train'

# --- 4. INITIALIZE & COMPILE ---
model = build_two_stream_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# --- 5. CREATE ACTUAL GENERATORS FOR TRAINING ---
train_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='training')
val_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='validation')

# --- 6. CALLBACKS ---
callbacks = [
    ModelCheckpoint('best_two_stream_model.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    CSVLogger('training_logs.csv', separator=',', append=False),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]


# --- 7. START TRAINING ---
print("\nStarting Day 3: Full Training Marathon...")

# We calculate these to avoid running out of data
# With 384 images and batch_size 32: 384 / 32 = 12 steps total
# Since we split 80/20: Train ~9 steps, Val ~3 steps
history = model.fit(
    train_gen,
    steps_per_epoch=9, 
    validation_data=val_gen,
    validation_steps=3,
    epochs=25,
    callbacks=callbacks
)

print("\nTraining Complete! Ready for Day 4 Evaluation.")