import cv2
from ultralytics import YOLO
import numpy as np
model=YOLO('bestCamaron.pt')
video_path = 'CamaronVideo.MP4'


output_path = 'output_video.avi'

cap = cv2.VideoCapture(video_path)


# Obtener propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


colores_etiquetas = {
    'aireadores': (255, 0, 0),  # Azul
    'alimentadores': (0, 255, 0),  # Verde
    'aireadores on': (0, 0, 255),  # Rojo
}
color = (0, 255, 0)
alpha = 0.3
box_color = (0, 0, 255)




name=["aireadores", "alimentadores", "aireadores on"]
object_counters = {}



if not cap.isOpened():
    print("Error al abrir el archivo de video.")

else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Realiza la predicción en el fotograma
        results = model.predict(frame, conf=0.2)

        # Crea una copia del fotograma para dibujar la sombra
        overlay = frame.copy()

        # Color de la sombra (verde en este caso) y nivel de transparencia
        color = (255, 0, 0)
        alpha = 0.3
        box_color = (255, 0, 0)
        current_frame_counters = {}
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Get box|es on CPU in numpy format
            for box in boxes:
                print(box.conf)
                print(name[int(box.cls.item())])
                print("------------------------uWu---------------------")
                print(current_frame_counters)
                # Extraer las coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                class_name = str(name[int(box.cls.item())])
                if class_name in current_frame_counters:
                    current_frame_counters[class_name] += 1
                else:
                    current_frame_counters[class_name] = 1

                    # Actualizar el contador total de objetos
                if class_name in object_counters:
                    object_counters[class_name] += 1
                else:
                    object_counters[class_name] = 1

                #Se define el contorno de las predicciones
                box_color = colores_etiquetas.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                etiqueta=name[int(box.cls.item())]
                porce=box.conf[0]
                # Mostrar el porcentaje de confianza y el nombre de la clase
                text = str(etiqueta)
                #text = str(etiqueta)+str(porce)

                #text = f"{box.conf[0]:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

                # Definir los puntos del contorno del polígono (en este caso, un rectángulo)
                vertices = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                # Dibujar un polígono relleno que represente la sombra
                cv2.fillPoly(overlay, [vertices], box_color)



            y_offset = 20  # Desplazamiento inicial en y para mostrar el texto
            for class_name, count in current_frame_counters.items():
                cv2.putText(frame, f"{class_name}: {count} --------", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 20


            # Combinar la capa de sombra con el fotograma original
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Mostrar el fotograma con las detecciones
            cv2.imshow('Detections with Shadow', frame)


        # Salir con 'ESC'
            if cv2.waitKey(1) == 27:
                break


cap.release()
out.release()
cv2.destroyAllWindows()