import os
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- CONFIGURACIÓN DE HARDWARE (RTX 2060 Super) ---
# Usamos FP16 (precisión mixta) para velocidad turbo y ahorro de VRAM
mixed_precision.set_global_policy('mixed_float16')

# --- HIPERPARÁMETROS ---
IMG_SIZE = 224
BATCH_SIZE = 32   # 32 es seguro. Si ves que le sobra VRAM, subelo a 64.
EPOCHS = 15       # Con 15 suele bastar para llegar a >95%
DATA_DIR = 'dataset_final'  # La carpeta que está creando tu otro script

def build_model():
    print("Construyendo arquitectura EfficientNetB0...")
    
    # 1. Capa de Aumento de Datos (Data Augmentation)
    # Ayuda a que el modelo generalice mejor y no memorice
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    # 2. Base del modelo (Cerebro pre-entrenado en ImageNet)
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelamos la base (no queremos romper lo que ya sabe EfficientNet)
    base_model.trainable = False 

    # 3. Ensamblaje
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)  # Aplicar variaciones
    x = base_model(x, training=False) # Pasar por EfficientNet
    x = layers.GlobalAveragePooling2D()(x) # Aplanar
    x = layers.Dropout(0.2)(x)  # Apagar 20% de neuronas al azar (evita overfitting)
    x = layers.Dense(128, activation='relu')(x)
    
    # Capa de salida (1 neurona: 0 o 1)
    # IMPORTANTE: dtype='float32' es obligatorio al final de mixed_precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = models.Model(inputs, outputs, name="DetectorIA_v1")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # --- 1. CARGAR DATOS ---
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No encuentro la carpeta '{DATA_DIR}'. Espera a que setup_dataset.py termine.")
        return

    print("Cargando datasets desde disco...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'train'),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, 'val'),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    # IMPRIMIR LAS CLASES (Ojo con esto para tu Bot)
    class_names = train_ds.class_names
    print(f"\n MAPEO DE CLASES DETECTADO: {class_names}")
    # Usualmente es alfabético: ['fake', 'real'] -> 0=Fake, 1=Real
    
    # Optimización de tubería de datos (Cache en RAM + Prefetch)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 2. ENTRENAMIENTO ---
    model = build_model()
    model.summary()
    
    # Callbacks (Los salvavidas)
    callbacks = [
        # Guardar solo si mejora la validación
        ModelCheckpoint(
            'models/best_model.keras', 
            save_best_only=True, 
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        # Parar si lleva 3 épocas sin mejorar (ahorra tiempo)
        EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        ),
        # Bajar velocidad de aprendizaje si se estanca
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=2, 
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n--- INICIANDO ENTRENAMIENTO ---")
    print("   (Presiona Ctrl+C si necesitas parar antes, se guardará el mejor checkpoint)")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    print("\n¡Entrenamiento finalizado!")
    print("   Modelo guardado en: models/best_model.keras")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    main()