import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
import traceback

# Configuration
IMG_SIZE = (224, 224)  # ResNet50 standard input size
BATCH_SIZE = 32  # Suitable for most systems
EPOCHS = 10  # Increased for better convergence
FINE_TUNE_EPOCHS = 10  # Extended fine-tuning
NUM_CLASSES = 8
DATA_PATH = r"C:\IEEE HACKATHON\GAN_generarated_images"
MODEL_PATH = 'best_model_optimized.h5'
WORKERS = 4  # Reduced to avoid overloading
SEED = 42
STEPS_PER_EPOCH = 600  # Fixed to 1000 as requested

def validate_dataset(data_path):
    """Validate dataset structure and return class names."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} does not exist")
    class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(class_names)}")
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            raise ValueError(f"No images found in class {class_name}")
        print(f"Class {class_name}: {len(images)} images")
    return class_names

def create_data_generators(data_path):
    """Create data generators for train and validation sets."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1,
        brightness_range=[0.9, 1.1]
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1
    )

    try:
        train_gen = train_datagen.flow_from_directory(
            data_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=SEED
        )

        val_gen = val_datagen.flow_from_directory(
            data_path,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=SEED
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create data generators: {str(e)}")

    return train_gen, val_gen

def evaluate_model(model, generator, class_names):
    """Evaluate model on the validation set."""
    try:
        y_true = generator.classes
        y_pred = model.predict(generator, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=class_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred_classes))
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

def main():
    try:
        # Set random seeds for reproducibility
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        print(f"TensorFlow Version: {tf.__version__}")
        print(f"GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        # Validate dataset
        class_names = validate_dataset(DATA_PATH)

        # Create data generators
        train_gen, val_gen = create_data_generators(DATA_PATH)
        print(f"Training samples: {train_gen.samples}, Classes: {train_gen.class_indices}")
        print(f"Validation samples: {val_gen.samples}")

        # Display class distribution
        class_counts = Counter(train_gen.classes)
        print("Training class distribution:")
        for idx, count in class_counts.items():
            print(f"{class_names[idx]}: {count} samples")

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
        )
        class_weights = dict(enumerate(class_weights))
        print(f"Class weights: {class_weights}")

        # Calculate validation steps
        VALIDATION_STEPS = max(1, val_gen.samples // BATCH_SIZE)  # Ensure at least 1 step
        print(f"Steps per epoch: {STEPS_PER_EPOCH}, Validation steps: {VALIDATION_STEPS}")

        # Build model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze base model initially
        base_model.trainable = False

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),  # Added gradient clipping
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_accuracy'),
            ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            TensorBoard(log_dir='./logs_optimized')
        ]

        # Initial training
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            workers=WORKERS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS
        )

        # Fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=1e-5, clipnorm=1.0),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        history_fine = model.fit(
            train_gen,
            epochs=FINE_TUNE_EPOCHS,
            initial_epoch=history.epoch[-1] + 1,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            workers=WORKERS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS
        )

        # Evaluate model
        print("\nEvaluating model on validation set:")
        evaluate_model(model, val_gen, class_names)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()