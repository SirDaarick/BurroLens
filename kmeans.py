import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

def kmeans_analizar_imagen(ruta_imagen, k=3, max_iter=20, tol=1.0):
    # Cargar imagen y convertir a RGB
    img = Image.open(ruta_imagen).convert("RGB")
    img_rgb = np.array(img)
    alto, ancho, _ = img_rgb.shape
    pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

    # Funciones internas
    def inicializar_centroides(pixels, k):
        indices = np.random.choice(len(pixels), k, replace=False)
        return pixels[indices]

    def asignar_clusters(pixels, centroides):
        distancias = np.linalg.norm(pixels[:, None] - centroides[None, :], axis=2)
        return np.argmin(distancias, axis=1)

    def actualizar_centroides(pixels, etiquetas, k):
        nuevos_centroides = np.zeros((k, pixels.shape[1]))
        for i in range(k):
            cluster_pixels = pixels[etiquetas == i]
            if len(cluster_pixels) > 0:
                nuevos_centroides[i] = cluster_pixels.mean(axis=0)
            else:
                nuevos_centroides[i] = pixels[np.random.choice(len(pixels))]
        return nuevos_centroides

    # Inicialización
    centroides = inicializar_centroides(pixels, k)
    etiquetas = asignar_clusters(pixels, centroides)

    # Iteraciones de K-means
    for _ in range(max_iter):
        nuevos_centroides = actualizar_centroides(pixels, etiquetas, k)
        if np.allclose(centroides, nuevos_centroides, atol=tol):
            break
        centroides = nuevos_centroides
        etiquetas = asignar_clusters(pixels, centroides)

    # Crear imagen segmentada
    seg_pixels = centroides[etiquetas.astype(int)]
    seg_img = seg_pixels.reshape((alto, ancho, 3)).astype(np.uint8)
    resultado_ruta = "resultado_kmeans.png"
    Image.fromarray(seg_img).save(resultado_ruta)

    # Crear resumen
    counts = np.bincount(etiquetas, minlength=k)
    total = np.sum(counts)
    resumen = f"Imagen segmentada en {k} clases (K-means):\n"
    for i in range(k):
        porcentaje = counts[i] / total * 100
        c = centroides[i].astype(int)
        resumen += f"Clase {i+1}: {counts[i]} píxeles ({porcentaje:.2f}%), color RGB={tuple(c)}\n"

    return resultado_ruta, resumen
