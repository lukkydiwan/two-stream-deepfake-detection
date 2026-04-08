import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from train import dual_generator , train_datagen , rgb_path , srm_path
# 1. LOAD THE DAY 3 MODEL
print("Loading the model from Day 3...")
model = load_model('best_two_stream_model.keras', compile=False)

# 2. UNFREEZE THE BACKBONE
# Find the EfficientNet layer (it's the 3rd layer in our functional graph)
effnet_layer = model.get_layer('efficientnetb0')
effnet_layer.trainable = True

# Fine-tuning: Freeze everything EXCEPT the last 30 layers
for layer in effnet_layer.layers[:-30]:
    layer.trainable = False

# 3. RE-COMPILE (CRITICAL: Use a very small learning rate)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 1e-5 is standard for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 4. SETUP NEW CALLBACKS
callbacks = [
    ModelCheckpoint('final_tuned_model.keras', monitor='val_auc', save_best_only=True, mode='max'),
    CSVLogger('fine_tuning_logs.csv', append=False)
]

train_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='training')
val_gen = dual_generator(rgb_path, srm_path, batch_size=32, subset='validation')


# 5. START FINE-TUNING
# Use the same train_gen and val_gen from your previous script
print("\nStarting Day 4: Fine-Tuning the SRM and RGB streams...")
model.fit(
    train_gen,
    steps_per_epoch=9,
    validation_data=val_gen,
    validation_steps=3,
    epochs=20,
    callbacks=callbacks
)