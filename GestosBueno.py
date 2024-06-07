from djitellopy import Tello
import cv2
import numpy as np
import mediapipe as mp
import math

# Initializing MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Frame source: 0-> from webcam 1-> from drone
frame_source = 0
in_speed = 50  # Valor de velocidad inicial
in_height = 50  # Valor de altura inicial

# Initializing camera stream
if frame_source == 0:
    capture = cv2.VideoCapture(0)
    drone = Tello()
    drone.connect()
    drone.streamoff()
    drone.streamon()
    


"""drone = Tello()
drone.connect()
drone.streamoff()
drone.streamon()"""

# Image size
h = 500
w = 500


def pulgar_arriba(landmarks):
    # Posiciones de los dedos
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

    # Distancias entre las yemas de los dedos
    dist_thumb_index = thumb_tip - index_tip
    dist_index_middle = index_tip - middle_tip

    # Comprobación de las condiciones
    thumb_above_others = thumb_tip < index_tip and thumb_tip < middle_tip
    dist_condition = dist_index_middle > dist_thumb_index

    return thumb_above_others and dist_condition


# Thumb down
def pulgar_abajo(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    thumb__mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y

    return (thumb__mcp < thumb_tip)


def dedo_indice(landmarks):
    # Posiciones del dedo índice
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_tip_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_pip_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    
    # Posiciones del dedo medio
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    
    # Posiciones del dedo anular
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    
    # Posiciones del dedo meñique
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    
    # Verificar que el dedo índice esté levantado
    index_up = index_tip < index_pip
    
    # Verificar que los otros dedos estén cerrados
    others_closed = (middle_tip > middle_pip) and (ring_tip > ring_pip) and (pinky_tip > pinky_pip)
    
    # Verificar que el dedo índice esté en posición vertical
    index_vertical = abs(index_tip_x - index_pip_x) < abs(index_tip - index_pip)
    
    return index_up and others_closed and index_vertical


def señal_paz(landmarks):
    # Posiciones del dedo índice
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    index_tip_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_pip_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
    
    # Posiciones del dedo medio
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    middle_tip_x = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    middle_pip_x = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
    
    # Posiciones del dedo anular
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    
    # Posiciones del dedo meñique
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    
    # Verificar que el dedo índice y el dedo medio estén levantados
    index_up = index_tip < index_pip
    middle_up = middle_tip < middle_pip
    
    # Verificar que los otros dedos estén cerrados
    others_closed = (ring_tip > ring_pip) and (pinky_tip > pinky_pip)
    
    # Verificar que los dedos índice y medio estén en posición vertical
    index_vertical = abs(index_tip_x - index_pip_x) < abs(index_tip - index_pip)
    middle_vertical = abs(middle_tip_x - middle_pip_x) < abs(middle_tip - middle_pip)
    
    return index_up and middle_up and others_closed and index_vertical and middle_vertical


def rockstar(landmarks):

    # Obtención de las posiciones Y de las yemas y las articulaciones PIP de los dedos
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP].y

    # Verificar que el dedo índice y el dedo meñique estén levantados
    index_up = index_tip < index_pip
    pinky_up = pinky_tip < pinky_pip

    # Verificar que los otros dedos (medio y anular) estén cerrados
    middle_closed = middle_tip > middle_pip
    ring_closed = ring_tip > ring_pip

    # Comprobación final para determinar si el gesto 'rockstar' está presente
    return index_up and pinky_up and middle_closed and ring_closed


def is_alien(landmarks):
    # Posiciones de los dedos
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
    ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])
    pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])
    ring_dip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y])
    thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
    thumb_cmc = np.array([landmarks[mp_hands.HandLandmark.THUMB_CMC].x, landmarks[mp_hands.HandLandmark.THUMB_CMC].y])

    # Distancias entre las yemas de los dedos
    dist_index_middle = np.linalg.norm(index_tip - middle_tip)
    dist_middle_ring = np.linalg.norm(middle_tip - ring_tip)
    dist_ring_pinky = np.linalg.norm(ring_dip - pinky_tip)

    # Condición para el pulgar horizontal
    thumb_horizontal = abs(thumb_tip[1] - thumb_cmc[1]) < abs(thumb_tip[0] - thumb_cmc[0])

    # Condición para la señal "alien"
    alien_signal = (dist_middle_ring > dist_index_middle) and (dist_ring_pinky < dist_middle_ring)

    return thumb_horizontal and alien_signal


def perfecto(landmarks):
    # Posiciones de los dedos
    index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
    thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
    ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])
    pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])

    # Distancias entre las puntas de los dedos
    dist_index_thumb = np.linalg.norm(index_tip - thumb_tip)
    dist_ring_pinky = np.linalg.norm(ring_tip - pinky_tip)

    # Comprobación de la condición modificada
    index_thumb_close = dist_index_thumb < dist_ring_pinky

    return index_thumb_close

def gira_D(landmarks):
    # Posiciones de los dedos
    wrist = landmarks[mp_hands.HandLandmark.WRIST].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].x
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x

    # Condiciones para detectar el dedo índice apuntando a la derecha
    index_extended = index_tip > index_mcp  # El índice está extendido
    other_fingers_contract = (
        middle_tip < index_mcp and
        ring_tip < index_mcp and
        pinky_tip < index_mcp and
        thumb_tip < index_mcp
    )  # Los otros dedos están contraídos

    # Condición para la muñeca en posición horizontal
    wrist_horizontal = landmarks[mp_hands.HandLandmark.WRIST].y

    return index_extended and other_fingers_contract and wrist_horizontal

def distancia_entre_puntos(p1, p2):
    return math.sqrt((p1.x - p2.x) * 2 + (p1.y - p2.y) * 2 + (p1.z - p2.z) ** 2)

def dedo_en_horizontal_y_apuntando_izquierda(landmark_tip, landmark_mcp):
    return landmark_tip.x < landmark_mcp.x and abs(landmark_tip.y - landmark_mcp.y) < abs(landmark_tip.x - landmark_mcp.x)

def muñeca_en_horizontal(landmark_wrist, landmark_thumb_cmc):
    return abs(landmark_wrist.y - landmark_thumb_cmc.y) < abs(landmark_wrist.x - landmark_thumb_cmc.x)

def dedo_en_vertical(landmark_tip, landmark_ip):
    return abs(landmark_tip.x - landmark_ip.x) < abs(landmark_tip.y - landmark_ip.y)

def rota_iz(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    # Condición para el dedo índice en posición horizontal y apuntando a la izquierda
    index_horizontal_left = dedo_en_horizontal_y_apuntando_izquierda(index_tip, index_mcp)
    
    # Condición para la muñeca en posición horizontal
    wrist_horizontal = muñeca_en_horizontal(wrist, thumb_cmc)
    
    # Condición para el pulgar en posición vertical y extendido
    thumb_vertical = dedo_en_vertical(thumb_tip, thumb_ip)
    thumb_extended = distancia_entre_puntos(thumb_tip, thumb_ip) > distancia_entre_puntos(thumb_ip, thumb_mcp)

    return index_horizontal_left and wrist_horizontal and thumb_vertical and thumb_extended


def distancia_entre_puntos(p1, p2):
    return math.sqrt((p1.x - p2.x) * 2 + (p1.y - p2.y) * 2 + (p1.z - p2.z) ** 2)

def dedos_en_vertical(landmarks, dedos):
    for dedo in dedos:
        tip = landmarks[mp_hands.HandLandmark[dedo + '_TIP']]
        pip = landmarks[mp_hands.HandLandmark[dedo + '_PIP']]
        if abs(tip.x - pip.x) >= abs(tip.y - pip.y):
            return False
    return True

def distancia_entre_puntos(p1, p2):
    return math.sqrt((p1.x - p2.x) * 2 + (p1.y - p2.y) * 2 + (p1.z - p2.z) ** 2)

def dedo_en_vertical(landmark_tip, landmark_pip):
    return abs(landmark_tip.x - landmark_pip.x) < abs(landmark_tip.y - landmark_pip.y)

def dedo_en_horizontal(landmark_tip, landmark_pip):
    return abs(landmark_tip.y - landmark_pip.y) < abs(landmark_tip.x - landmark_pip.x)

def tres(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]

    # Condiciones de distancia
    dist_thumb_index = distancia_entre_puntos(thumb_tip, index_tip)
    dist_middle_ring = distancia_entre_puntos(middle_tip, ring_tip)
    dist_pinky_ring = distancia_entre_puntos(pinky_tip, ring_tip)

    condition1 = dist_thumb_index > dist_middle_ring
    condition2 = dist_pinky_ring > dist_middle_ring

    # Condición de posición del pulgar en horizontal
    thumb_horizontal = dedo_en_horizontal(thumb_tip, thumb_ip)

    # Condiciones de posición de los otros dedos en vertical
    index_vertical = dedo_en_vertical(index_tip, index_pip)
    middle_vertical = dedo_en_vertical(middle_tip, middle_pip)
    ring_vertical = dedo_en_vertical(ring_tip, ring_pip)
    pinky_vertical = dedo_en_vertical(pinky_tip, pinky_pip)

    return condition1 and condition2 and thumb_horizontal and index_vertical and middle_vertical and ring_vertical and pinky_vertical

def main():

    global is_flying # Utiliza la variable global
    is_flying = False


    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        print("main program running now")
        # Main cycle
        while True:
            # Obtaining a new frame
            if frame_source == 0:
                ret, img = capture.read()
                if not ret:
                    print("Failed to capture image")
                    continue
            elif frame_source == 1:
                frame_read = drone.get_frame_read()
                img = frame_read.frame
                # Going from RGB to BGR color workspace
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Rotating the image
            img = cv2.flip(img, 1)

            # Resizing the image --- cv2.resize('ImageName',(x_dimension,y_dimension))
            img = cv2.resize(img, (500, 500))
            
            # hand detection
            results = hands.process(img)

            # Variable para rastrear si se ha detectado el gesto
            thumb_up_detected = False
            thumb_down_detected = False
               

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # drwing landmarks and connections
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Verifica si el pulgar esta arriba
                    if pulgar_arriba(hand_landmarks.landmark):
                        cv2.putText(img, "Despega", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        thumb_up_detected = True  # Marcar que se ha detectado el gesto de pulgar arriba

                        if not is_flying:
                            drone.takeoff()
                            is_flying = True

                    # Verifica si el pulgar está abajo
                    if pulgar_abajo(hand_landmarks.landmark):
                        cv2.putText(img, "Aterriza", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        thumb_down_detected = True  # Marcar que se ha detectado el gesto de pulgar abajo

                    if dedo_indice(hand_landmarks.landmark):
                        cv2.putText(img, "Avanza enfrente", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, 50, 0, 0)

                    if señal_paz(hand_landmarks.landmark):
                        cv2.putText(img, "Avanza Atras", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, -50, 0, 0)
                    
                    if rockstar(hand_landmarks.landmark):
                        cv2.putText(img, "Sube", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, 0, 50, 0)
                    
                    if is_alien(hand_landmarks.landmark):
                        cv2.putText(img, "Baja", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, 0, -50, 0)

                    if perfecto(hand_landmarks.landmark):
                        cv2.putText(img, "Avanza Izquierda", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(-50, 0, 0, 0)
                    
                    if tres(hand_landmarks.landmark):
                        cv2.putText(img, "Avanza Derecha", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(-50, 0, 0, 0)

                    if gira_D(hand_landmarks.landmark):
                        cv2.putText(img, "rota derecha", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, 0, 0, 50)

                    if rota_iz(hand_landmarks.landmark):
                        cv2.putText(img, "rota izquierda", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        drone.send_rc_control(0, 0, 0, -50)

                    

            if thumb_down_detected and is_flying:
                drone.land()
                is_flying = False
            
                    

            # Writing the battery level in the image cv2.putText(ImageName, text, location, font, scale, color, thickness)
            if frame_source == 1:
                cv2.putText(img, 'Battery:  ' + str(drone.get_battery()), (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0),3)

            # Showing the image in a window
            cv2.imshow("Hand Gesture Recognition", img)

            # Keyboard monitor
            key = cv2.waitKey(1) & 0xFF

            # close the windows and break the program if 'q' is pressed
            if key == 113:
                cv2.destroyAllWindows()
                if frame_source == 1:
                    # drone.land()
                    drone.streamoff()
                    drone.end()
                break


try:
    main()

except KeyboardInterrupt:
    print('KeyboardInterrupt exception is caught')
    cv2.destroyAllWindows()
    if frame_source == 1:
        drone.land()
        drone.streamoff()
        drone.end()

else:
    print('No exceptions are caught')