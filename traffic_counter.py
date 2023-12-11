import numpy as np
import cv2
from tracker import *

#Remove componentes pequenos de uma imagem binária.
def removesmall(img, min_size):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    #your answer image
    img2 = np.zeros((output.shape), dtype=np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    return img2

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


#Cria uma instância do objeto do arquivo tracker.py:
# - a primeira tupla define a velocidade esperada,
# - o segundo valor define o raio de busca para rastreamento.
tracker = EuclideanDistTracker((0,15.0), 45.0)

cap = cv2.VideoCapture("Videos/road_traffic_4.mkv")

#Cria o objeto para subtração de fundo.
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
#object_detector = cv2.createBackgroundSubtractorKNN()


# Define the positions of the two lines
line_up = 300
line_down = 300

# Initialize counters
count_up = 0
count_down = 0

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
    #roi = vis[250:720, 170:948]
    roi = vis[500:2000, 1000:2000]

    #Calcula a máscara por subtração de fundo.
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    mask = removesmall(mask, 80)

    #Aplica operadores de Morfologia Matemática da erosão e dilatação,
    #para filtrar regiões indesejadas e juntar componentes próximos.
    kernel = circularkernel(30)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    #Para extrair os contornos dos componentes.
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        #Elimina contornos de objetos com pequena área ou razão de aspecto inesperada.
        if area > 300 and (area > 0.5*w*h) and (1/5.0 < w/h < 5.0):
            detections.append([x,y,w,h])
    center_points = tracker.getcenterpoints()

    #Atualiza o tracker para o rastreamento.
    boxes_ids = tracker.update(detections)
    # Initialize counters

    for box_id in boxes_ids:
        x, y, w, h, vid = box_id
        cv2.putText(roi, str(vid), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ymid = y + h // 2
        # Procura objeto de mesmo id no frame anterior.
        if vid in center_points.keys():
            _, cy = center_points[vid]
            # Check the direction of movement based on centroid position
            if ymid > line_down and cy <= line_down:
                count_down += 1
            elif ymid < line_up and cy >= line_up:
                count_up += 1
            

    #Desenha a linha de chegada.
    cv2.line(roi, (0, line_up), (1000, line_up), (0, 0, 255), 2)
    cv2.line(roi, (0, line_down), (1000, line_down), (0, 0, 255), 2)
    cv2.putText(roi, 'Vehicles Up: %d' % count_up, (0, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(roi, 'Vehicles Down: %d' % count_down, (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    cv2.putText(roi, 'Total Vehicles: %d' % (count_up + count_down) , (0, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    #Mostra o valor atual do contador de veículos.
    
    cv2.imshow("Roi", roi)
    cv2.imshow("Mask", mask)

    if f == 1:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(30)

    if key == 27:
        break

#print("Vehicles: %d"%count)
cap.release()
cv2.destroyAllWindows()
