import numpy as np
import cv2
from tracker import *

#Cria um elemento estruturante circular de raio d.
def circularkernel(d):
    y,x = np.ogrid[-d: d+1, -d: d+1]
    kernel = x**2+y**2 <= d**2
    kernel = kernel.astype(np.uint8)
    return kernel

#Cria um elemento estruturante de formato elíptico.
#Similar ao resultado da função:
#cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dx*2+1,dy*2+1))
def ellipticalkernel(dx, dy):
    y,x = np.ogrid[-dy: dy+1, -dx: dx+1]
    kernel = x**2/dx**2+y**2/dy**2 <= 1.0
    kernel = kernel.astype(np.uint8)
    return kernel

#variável global para contagem dos veículos.
count = 0
#Cria uma instância do objeto do arquivo tracker.py:
# - a primeira tupla define a velocidade esperada,
# - o segundo valor define o raio de busca para rastreamento.
tracker = EuclideanDistTracker((0,15.0), 20.0)

cap = cv2.VideoCapture("./road_traffic_1.mp4")

#Cria o objeto para subtração de fundo.
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
#object_detector = cv2.createBackgroundSubtractorKNN()

f = 0
while True:
    f += 1
    #Lê o próximo frame do vídeo.
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape
    #print(height, width)
    vis = frame.copy()

    #Define a região de interesse (ROI - region of interest).
    roi = vis[340:720, 480:810]

    #Calcula a máscara por subtração de fundo.
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    #Aplica operadores de Morfologia Matemática da erosão e dilatação,
    #para filtrar regiões indesejadas e juntar componentes próximos.
    kernel = circularkernel(1)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    kernel = circularkernel(6)
    erosion = cv2.erode(mask,kernel,iterations = 1)

    kernel = ellipticalkernel(6, 9)
    opening = cv2.dilate(erosion,kernel,iterations = 1)
    mask = opening

    #Para extrair os contornos dos componentes.
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        #Elimina contornos de objetos com pequena área ou razão de aspecto inesperada.
        if area > 100 and (area > 0.5*w*h) and (1/5.0 < w/h < 5.0):
            detections.append([x,y,w,h])
    center_points = tracker.getcenterpoints()

    #Atualiza o tracker para o rastreamento.
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,vid = box_id
        cv2.putText(roi, str(vid), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 3)
        ymid = y+h//2
        #Procura objeto de mesmo id no frame anterior.
        if vid in center_points.keys():
            _,cy = center_points[vid]
            #Compara a posição atual do objeto com sua posição no
            #frame anterior para saber se cruzou a linha de chegada.
            if ymid >= 290 and cy < 290:
                count += 1

    #Desenha a linha de chegada.
    cv2.line(roi, (0,290), (329,290), (0,0,255), 2)
    #Mostra o valor atual do contador de veículos.
    cv2.putText(roi, 'Vehicles: %d'%count, (0,25), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    cv2.imshow("Roi", roi)
    cv2.imshow("Mask", mask)

    if f == 1:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(30)

    if key == 27:
        break

print("Vehicles: %d"%count)
cap.release()
cv2.destroyAllWindows()
