import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set memory-efficient policy
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Paths
project_path = "/Users/mananmathur/Documents/Academics/MIT/subject matter/YEAR 4/SEM 8/PROJECT/project"
base_path = os.path.join(project_path, "OCTID")
model_save_dir = os.path.join(project_path, "logs/models")
training_log_dir = os.path.join(project_path, "logs/training")
result_path = os.path.join(project_path, "logs/results/customCNN_OCTID_KFold.xlsx")
image_size = 224
batch_size = 16
num_classes = 5
epochs = 20
finetune_epochs = 10
n_splits = 5

# CLAHE Preprocessor
def clahe_preprocess(img):
    img = img.numpy() if isinstance(img, tf.Tensor) else img
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return img

def custom_preprocessing(image):
    image = tf.numpy_function(func=clahe_preprocess, inp=[image], Tout=tf.uint8)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

# Load image paths + labels
data = []
for label, folder in enumerate(sorted(os.listdir(os.path.join(base_path, "Train")))):
    folder_path = os.path.join(base_path, "Train", folder)
    if not os.path.isdir(folder_path):
        continue
    for fname in os.listdir(folder_path):
        data.append((os.path.join(folder_path, fname), folder))

data = pd.DataFrame(data, columns=['path', 'label'])

# Optional hold-out test set (10%)
data, test_df = train_test_split(data, stratify=data['label'], test_size=0.1, random_state=42)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(data['path'], data['label']), 1):
    print(f"\n===== Fold {fold} =====")
    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()

    train_gen = ImageDataGenerator(
        preprocessing_function=lambda x: custom_preprocessing(tf.cast(x, tf.uint8)),
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_gen = ImageDataGenerator(preprocessing_function=lambda x: custom_preprocessing(tf.cast(x, tf.uint8)))

    train_data = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_data = val_gen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    label_map = train_data.class_indices
    class_weights = compute_class_weight('balanced', classes=np.array(list(label_map.values())), y=train_df['label'].map(label_map))
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = Sequential([
        Rescaling(1./255, input_shape=(image_size, image_size, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(256, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])

    model.compile(
        optimizer=Adam(learning_rate=3e-4, clipvalue=1.0),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    def cosine_annealing(epoch):
        lr_max = 3e-4
        lr_min = 1e-6
        return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs)) / 2

    callbacks = [
        ModelCheckpoint(os.path.join(model_save_dir, f"customCNN_OCTID_fold{fold}.keras"), monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        LearningRateScheduler(cosine_annealing),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
        CSVLogger(os.path.join(training_log_dir, f"customCNN_OCTID_fold{fold}.csv"), append=True)
    ]

    model.fit(train_data, validation_data=val_data, epochs=epochs, class_weight=class_weight_dict, callbacks=callbacks)

    scores = model.evaluate(val_data, verbose=0)
    y_true = [k for k in val_data.classes]
    inv_map = {v: k for k, v in val_data.class_indices.items()}
    y_true_labels = [inv_map[i] for i in y_true]
    y_pred_labels = [inv_map[i] for i in np.argmax(model.predict(val_data), axis=1)]
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)

    result = {
        "Fold": fold,
        "Test Accuracy": scores[1],
        "Test Loss": scores[0],
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score']
    }
    fold_results.append(result)

# Final test evaluation (optional)
test_gen = ImageDataGenerator(preprocessing_function=lambda x: custom_preprocessing(tf.cast(x, tf.uint8)))
test_data = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

best_model_path = os.path.join(model_save_dir, "customCNN_OCTID_fold1.keras")
model.load_weights(best_model_path)
test_scores = model.evaluate(test_data, verbose=1)
test_report = classification_report(test_data.classes, np.argmax(model.predict(test_data), axis=1), output_dict=True)

print("\n✅ Final Test Evaluation:")
print(f"Accuracy: {test_scores[1]:.4f}, Loss: {test_scores[0]:.4f}")

results_df = pd.DataFrame(fold_results)
results_df.to_excel(result_path, index=False)
print("✅ Custom CNN KFold results saved.")
