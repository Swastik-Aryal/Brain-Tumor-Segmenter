from utils import *
from datagen import DataGenerator, augmentation
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.model_selection import train_test_split

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm


metrics = [
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(),
    dice_coefficient_necrotic,
    dice_coefficient_edema,
    dice_coefficient_enhancing,
]

logger = CSVLogger("/kaggle/working/logs.csv", separator=",", append=False)
callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.000001, verbose=1
    ),
    EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
    ),
    logger,
]


# Model Definition
BACKBONE = "resnet50"

model = sm.Unet(
    BACKBONE,
    classes=4,
    activation="softmax",
    encoder_weights=None,
    input_shape=(128, 128, 4),
)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=metrics,
)


# Split the data
list_ids = list(range(len(os.listdir("dataset/flair"))))
train_ids, valid_ids = train_test_split(list_ids, test_size=0.4, random_state=42)
valid_ids, test_ids = train_test_split(valid_ids, test_size=0.5, random_state=42)

print(f"Train : {len(train_ids)} Valid : {len(valid_ids)} Test : {len(test_ids)}")

train_generator = DataGenerator(train_ids, augmentation=augmentation)
valid_generator = DataGenerator(valid_ids, augmentation=None)
test_generator = DataGenerator(test_ids, augmentation=None)

# Train the model
history = model.fit(
    train_generator, validation_data=valid_generator, epochs=100, callbacks=callbacks
)

# Save the model
model.save("TumorSegmentation.h5")
