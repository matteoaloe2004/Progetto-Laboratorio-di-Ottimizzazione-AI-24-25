from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(64,64,3), num_classes=38):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_transfer_model(input_shape=(128,128,3), num_classes=38, fine_tune_at=100):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

    # 🔒 Fissa tutti i layer prima di `fine_tune_at`
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 🔓 Sblocca i layer da `fine_tune_at` in poi
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # learning rate più basso per evitare "catastrofi"
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
