import cv2
import mediapipe as mp
import numpy as np

# ---------- CONFIGURACIÓN ----------
WIDTH, HEIGHT = 1280, 720
FIG_SIZE = 120                      # <<<<< MÁS GRANDES
COLOR_SQR = (255, 0, 0)
COLOR_TRI = (0, 255, 0)
COLOR_CIR = (0, 0, 255)

# Posiciones iniciales (columna izquierda)
figs = {
    'square':   {'xy': (50, 100),  'color': COLOR_SQR, 'grabbed': False},
    'triangle': {'xy': (50, 280),  'color': COLOR_TRI, 'grabbed': False},
    'circle':   {'xy': (50, 460),  'color': COLOR_CIR, 'grabbed': False}
}

# ---------- FUNCIONES ----------
def draw_figures(img):
    for name, f in figs.items():
        x, y = f['xy']
        color = (0, 255, 255) if f['grabbed'] else f['color']
        if name == 'square':
            cv2.rectangle(img, (x, y), (x+FIG_SIZE, y+FIG_SIZE), color, -1)
        elif name == 'triangle':
            pts = np.array([[x+FIG_SIZE//2, y], [x, y+FIG_SIZE], [x+FIG_SIZE, y+FIG_SIZE]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif name == 'circle':
            cv2.circle(img, (x+FIG_SIZE//2, y+FIG_SIZE//2), FIG_SIZE//2, color, -1)

def point_in_rect(px, py, x, y, w, h):
    return x < px < x+w and y < py < y+h

def point_in_circle(px, py, cx, cy, r):
    return (px-cx)**2 + (py-cy)**2 < r*r

def point_in_triangle(px, py, tri_pts):
    v0 = tri_pts[2] - tri_pts[0]
    v1 = tri_pts[1] - tri_pts[0]
    v2 = np.array([px, py]) - tri_pts[0]
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v < 1)

# ---------- MEDIAPIPE ----------
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# ---------- CÁMARA ----------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    draw_figures(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]

            thumb_tip = lm[4]
            index_tip = lm[8]

            # Centro entre pulgar e índice
            cx, cy = (thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2

            # Pinza cerrada solo si distancia pulgar-índice es pequeña
            pinza_cerrada = np.hypot(thumb_tip[0]-index_tip[0], thumb_tip[1]-index_tip[1]) < 50

            # ¿Sobre qué figura?
            for name, f in figs.items():
                x, y = f['xy']
                if name == 'square' and point_in_rect(cx, cy, x, y, FIG_SIZE, FIG_SIZE):
                    fig_obj = name; break
                elif name == 'circle' and point_in_circle(cx, cy, x+FIG_SIZE//2, y+FIG_SIZE//2, FIG_SIZE//2):
                    fig_obj = name; break
                elif name == 'triangle':
                    tri = np.array([[x+FIG_SIZE//2, y], [x, y+FIG_SIZE], [x+FIG_SIZE, y+FIG_SIZE]], np.int32)
                    if point_in_triangle(cx, cy, tri):
                        fig_obj = name; break
            else:
                fig_obj = None

            # Agarrar / soltar
            if pinza_cerrada and fig_obj and not figs[fig_obj]['grabbed']:
                figs[fig_obj]['grabbed'] = True
            elif not pinza_cerrada:
                for f in figs.values():
                    f['grabbed'] = False

            # Mover figura agarrada
            for name, f in figs.items():
                if f['grabbed']:
                    f['xy'] = (cx - FIG_SIZE//2, cy - FIG_SIZE//2)

    cv2.imshow("Mover figuras con pinza (pulgar+índice)", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()