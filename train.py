from model import create_cnn_model, create_transfer_model
from augmentation import get_data_augmentation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm.keras import TqdmCallback

def train(data_dir, img_size=(224, 224), batch_size=32, epochs=10, use_transfer_learning=False):
    datagen = get_data_augmentation()

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=42
    )

    num_classes = train_generator.num_classes
    input_shape = img_size + (3,)

    model = create_transfer_model(input_shape, num_classes) if use_transfer_learning else create_cnn_model(input_shape, num_classes)

    callbacks = [
        ModelCheckpoint('saved_models/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        TqdmCallback(verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        verbose=0
    )

    return model, history, val_generator
