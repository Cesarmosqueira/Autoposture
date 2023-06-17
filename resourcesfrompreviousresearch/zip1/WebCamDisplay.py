import cv2
import DetectPoseFunction as dp

def webCamDisplay():
    
    # Configurar la función Pose para el vídeo.
    pose_video = dp.mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, model_complexity = 1)

    # Inicializa el objeto VideoCapture para leer de la webcam.
    video = cv2.VideoCapture(0)

    # Crear una ventana con nombre para cambiar el tamaño
    cv2.namedWindow('Pose Detection in 3D', cv2.WINDOW_NORMAL)

    # Seteando las dimensiones de la camara
    video.set(3,1280)
    video.set(4,960)

    while video.isOpened():

        # Leer un frame
        ok, frame = video.read()

        # Si el frame no esta Ok
        if not ok:
            break
        
        # Voltea el marco horizontalmente para una visualización natural (selfie-view).
        frame = cv2.flip(frame, 1)

        # Obtener la anchura y la altura del frame
        frame_height, frame_width, _ =  frame.shape

        # Cambia el tamaño del cuadro manteniendo la relación de aspecto.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # Realice la detección de puntos de referencia de la pose.
        frame, _ = dp.detectPose(frame, pose_video, display = True, verbose=True)

        # Mostrar el frame
        cv2.imshow('Pose Detection in 3D', frame)

        # capturar que tecla se presiona
        k = cv2.waitKey(1) & 0xFF

        # si pulsa 'ESC' cerrar ventana
        if(k == 27):
            break

    # Liberar el objeto VideoCapture.
    video.release()

    # Cerrar la ventana
    cv2.destroyAllWindows()

webCamDisplay()
