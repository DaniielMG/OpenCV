import cv2
import numpy as np

# Función para obtener el centro de un rectángulo (x, y, w, h)
def get_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

# Función para encontrar el tracker más cercano a una detección
def find_closest_tracker(detection, trackers, threshold=60):
    x, y, w, h = detection
    detection_center = get_center((x, y, w, h))

    closest_tracker = None
    min_distance = threshold

    for tracker in trackers:
        tracker_center = get_center(tracker.last_seen)
        distance = np.linalg.norm(np.array(detection_center) - np.array(tracker_center))

        if distance < min_distance:
            closest_tracker = tracker
            min_distance = distance

    return closest_tracker

# Clase para el seguimiento de personas
class PersonTracker:
    def __init__(self, initial_detection, color, frame, id):
        x, y, w, h = initial_detection
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, initial_detection)
        self.color = color
        self.last_seen = initial_detection
        self.id = id
        self.failed_updates = 0
        
    def update(self, frame):
        success, box = self.tracker.update(frame)
        if success:
            self.last_seen = box
            self.failed_updates = 0
        else:
            self.failed_updates += 1
        return success, box

# Cargar el clasificador Haar Cascade para la detección de cuerpos completos
haar_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicializar la captura de video desde un archivo
cap = cv2.VideoCapture("people_walking2.mp4")
trackers = []
next_id = 0

# Función para generar IDs únicos
def generate_unique_id():
    global next_id
    result = next_id
    next_id += 1
    return result

# Escalar el factor de reducción de la imagen y el intervalo de detección
scale_factor = 0.5
detection_interval = 3
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % detection_interval == 0:
        # Detectar cuerpos completos en escala de grises
        bodies = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Iterar sobre cada cuerpo detectado
        for (x, y, w, h) in bodies:
            if w > 200 or h > 200:  # Filtrar detecciones demasiado grandes
                continue

            closest_tracker = find_closest_tracker((x, y, w, h), trackers)

            if closest_tracker is None:
                new_tracker = PersonTracker((x, y, w, h), (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255))), frame, generate_unique_id())
                trackers.append(new_tracker)

    # Actualizar y dibujar los trackers existentes
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in tracker.last_seen]
            cv2.rectangle(frame, (x, y), (x+w, y+h), tracker.color, 2)

    # Eliminar trackers que han fallado más de 5 veces
    trackers = [tracker for tracker in trackers if tracker.failed_updates < 5]

    # Mostrar el frame con los rectángulos dibujados
    cv2.imshow('Frame', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
