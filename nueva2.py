import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Crear un generador de imágenes
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Directorio de las imágenes
image_dir = 'data\Entrenamineto\cuchara'  # Reemplaza esto con la ruta a tu directorio de imágenes

# Crear el directorio 'preview' si no existe
if not os.path.exists('preview1'):
    os.makedirs('preview1')

# Leer todas las imágenes en el directorio
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        img = load_img(image_path)  # Cargar la imagen
        x = img_to_array(img)  # Convertir la imagen en un array
        x = x.reshape((1,) + x.shape)  # Añadir una dimensión extra

        # Guardar 15 imágenes generadas para cada imagen
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='preview1', save_prefix=filename, save_format='jpeg'):
            i += 1
            if i > 5:  # Guardar 15 imágenes y luego detenerse
                break