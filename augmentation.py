from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255,
        validation_split=0.2  # se serve per splitting
    )
