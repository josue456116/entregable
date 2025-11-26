import cv2
import numpy as np

def detectar_figuras(img):

    # ============================
    #   CORRECCIÓN IMPORTANTE
    # ============================
    # Si la imagen viene del canvas tendrá 4 canales (RGBA)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Copia para dibujar
    salida = img.copy()

    # --- Preprocesamiento ---
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (7, 7), 2)

    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(approx)

        figura = "Desconocida"

        circ = (4 * np.pi * area) / (peri * peri)
        if circ > 0.78:
            figura = "Circulo"

        elif vertices == 3:
            figura = "Triangulo"

        elif vertices == 4:
            ratio = w / float(h)

            lados = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                lados.append(np.linalg.norm(p1 - p2))
            lados = np.array(lados)

            if np.std(lados) < 8 and 0.90 < ratio < 1.10:
                figura = "Cuadrado"

            elif np.std(lados) < 8:
                figura = "Rombo"

            elif abs(lados[0] - lados[2]) < 10 and abs(lados[1] - lados[3]) < 10:
                figura = "Paralelogramo"

            elif 0.50 < ratio < 2.0:
                figura = "Rectangulo"

            else:
                figura = "Trapecio"

        elif vertices == 5:
            figura = "Pentagono"

        elif vertices == 6:
            figura = "Hexagono"

        elif vertices >= 7:
            angulos = []
            for i in range(vertices):
                p1 = approx[i - 1][0]
                p2 = approx[i][0]
                p3 = approx[(i + 1) % vertices][0]

                v1 = p1 - p2
                v2 = p3 - p2

                ang = np.degrees(
                    np.arccos(np.dot(v1, v2) /
                              (np.linalg.norm(v1) * np.linalg.norm(v2)))
                )
                angulos.append(ang)

            if np.sum(np.array(angulos) < 60) >= vertices * 0.35:
                figura = "Estrella"
            else:
                figura = "Poligono"

        cv2.drawContours(salida, [approx], -1, (0, 255, 0), 3)
        cv2.putText(salida, figura, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return salida
