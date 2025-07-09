import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_plantvillage_data(data_dir, img_size=(64,64), batch_size=32, val_split=0.2, seed=42):
    """
    Carica dataset PlantVillage da directory strutturata:
    data_dir/
       healthy/
       disease1/
       disease2/
       ...
    
    Restituisce generatori train e validation con splitting automatico.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed,
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        shuffle=True
    )

    return train_generator, val_generator
