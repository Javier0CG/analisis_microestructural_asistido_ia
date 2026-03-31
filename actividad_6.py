!uv pip install ultralytics
import ultralytics
ultralytics.checks()

from ultralytics import SAM

# cargar el modelo
# opciones: SAM=sam_b.pt, SAM2=sam2_b.pt, SAM2.1=sam2.1_b.pt
model = SAM("sam2.1_b.pt")

# mostrar la información del modelo
model.info()

# segmentar una imagen
results = model("/content/nodulos_grafito.png")

# mostrar los resultados
results[0].show()