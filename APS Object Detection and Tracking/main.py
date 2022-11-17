import cv2
from tracker import *

capt = cv2.VideoCapture("rodovia.mp4")

tracker = EuclideanDistTracker()

# Encontrar objetos se movendo em fundo estatico

encontrar_objeto = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, quad = capt.read()
    height, width, _ = quad.shape

    # Mostrar apenas a area de interesse

    roi = quad[200: 700,500: 1000]

    mascara = encontrar_objeto.apply(roi)
    _, mascara = cv2.threshold(mascara, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calcular a area e retirar objetos indesejaveis
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

# Rastreamento do objeto

    caixas_ids = tracker.update(detections)
    for caixas_ids in caixas_ids:
        x, y, w, h, id = caixas_ids
        cv2.putText(roi, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Quadro", quad)
    cv2.imshow("Mascara", mascara)

    key = cv2.waitKey(30)
    if key == 27:
        break

capt.release()
cv2.destroyAllWindows()

