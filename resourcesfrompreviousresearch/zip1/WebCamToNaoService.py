import cv2
import DetectPoseFunction as dp
import getAnglesFromPoints as geom
import time

import socket

def gen_dict(joints, angles):
    return { x:y for x,y in zip(joints, angles) }

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect(('192.168.1.13', 9559))

def send(msg):
    print('Sending message:')
    print('\t' + msg.decode('ascii'))
    totalsent = 0
    while totalsent < len(msg):
        sent = s.send(msg[totalsent:])
        if sent == 0:
            print('connection broken')
            break
        totalsent += sent

import json

def webCamToNaoService():
    
    # Configurar la función Pose para el vídeo.
    pose_video = dp.mp_pose.Pose(static_image_mode = False, min_detection_confidence = 0.5, model_complexity = 1)

    # Inicializa el objeto VideoCapture para leer de la webcam.
    video = cv2.VideoCapture(0)

    # Crear una ventana con nombre para cambiar el tamaño
    cv2.namedWindow('Pose Detection in 3D', cv2.WINDOW_NORMAL)

    # Seteando las dimensiones de la camara
    video.set(3,1280)
    video.set(4,960)

    frame_count = 0
    start_time = time.time()
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
        frame, landmarks = dp.detectPose(frame, pose_video, display = False, verbose=False)

        # Mostrar el frame
        cv2.imshow('Pose Detection in 3D', frame)
        if False: #len(landmarks):
            angles = geom.getAnglesFromResults(landmarks)
            print(angles)
            joint_dict = gen_dict(geom.names, angles)
            # send(json.dumps(joint_dict).encode('ascii'))
            time.sleep(0.2)

        frame_count += 1
    
        # Calculate and display FPS every second
        if time.time() - start_time >= 1:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
            
            # Reset variables for the next calculation
            frame_count = 0
            start_time = time.time()

        # capturar que tecla se presiona
        k = cv2.waitKey(1) & 0xFF

        # si pulsa 'ESC' cerrar ventana
        if(k == ord('q')):
            break


    # Liberar el objeto VideoCapture.
    video.release()

    # Cerrar la ventana
    cv2.destroyAllWindows()

webCamToNaoService()
