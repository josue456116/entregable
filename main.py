from flask import Flask, render_template, request, Response
import cv2
import os
from werkzeug.utils import secure_filename
from detector import detectar_figuras

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cámara
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detectar_figuras(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":

        # VALIDAR que el input "imagen" realmente llegó
        if "imagen" not in request.files:
            print("⚠ No llegó ninguna imagen desde el canvas.")
            return render_template("index.html", output_image=None)

        archivo = request.files["imagen"]

        # VALIDAR archivo vacío
        if archivo.filename == "":
            print("⚠ Archivo vacío recibido.")
            return render_template("index.html", output_image=None)

        # Guardar siempre como la misma imagen
        ruta_imagen = os.path.join(app.config["UPLOAD_FOLDER"], "entrada_canvas.png")
        archivo.save(ruta_imagen)

        # Cargar imagen
        img = cv2.imread(ruta_imagen)

        if img is None:
            print("❌ Error: cv2 no pudo leer la imagen.")
            return render_template("index.html", output_image=None)

        # Procesar
        procesada = detectar_figuras(img)

        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")
        cv2.imwrite(output_path, procesada)

        output_image = "/static/output.jpg"

    return render_template("index.html", output_image=output_image)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
