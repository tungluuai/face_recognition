import argparse
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from keras_facenet import FaceNet
import cv2
import pickle
import numpy as np

def crop_image(original_image, coordinate):
    cropped_image = original_image[int(coordinate[1]):int(coordinate[3]), int(coordinate[0]):int(coordinate[2])]
    cropped_image = cv2.resize(cropped_image, (160,160))
    return cropped_image

class FaceRecognitionApp:
    def __init__(self, root,YOLO_model, embedder, SVM_model):
        self.root = root
        self.detection_model = YOLO_model
        self.face_embedding = embedder
        self.recognition_model = SVM_model
        self.all_names = list(np.load('../../names/all_names.npy').tolist())
        self.root.title("Face Recognition App")

        self.button_browse = tk.Button(root, text="Browse Image", command=self.browse_image, font=("Helvetica", 12), padx=10, pady=5, bg="#4CAF50", fg="white")
        self.button_browse.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.label_image = tk.Label(root)
        self.label_image.grid(row=0, column=1, padx=20, pady=20, sticky="e")

        self.label_name = tk.Label(root, text="Predicted Name: ", font=("Helvetica", 14), pady=10)
        self.label_name.grid(row=1, column=1, padx=20, pady=10, sticky="e")

    def browse_image(self):
        file_path = filedialog.askopenfilename()

        if file_path:
            # Dự đoán tên từ ảnh
            predicted_name, prob = self.predict(file_path)

            # Hiển thị ảnh và tên dự đoán
            self.display_image(file_path, predicted_name,prob)

    def predict(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            return "Unknown", None
        
        face = self.detection_model(img, conf = 0.5)
        coordinate = face[0].cpu().boxes.xyxy.tolist()
        if len(coordinate) == 1:
            [x1, y1, x2, y2] = coordinate[0]
            cropped_image = crop_image(img, [x1, y1, x2, y2])
            samples = np.expand_dims(cropped_image, axis=0)
            vector_embedding = self.face_embedding.embeddings(samples)
            yhat_prob = self.recognition_model.predict_proba(vector_embedding)
            if np.max(yhat_prob) > 0.6:
                return self.all_names[np.argmax(yhat_prob, axis=1)[0]], np.max(yhat_prob)
        return "Unknown", None

    def display_image(self, image_path, predicted_name,prob):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((400, 400))
        image = ImageTk.PhotoImage(image)

        self.label_image.config(image=image)
        self.label_image.image = image
        if prob is None:
            self.label_name.config(text=f"Predicted Name: {predicted_name}", font=("Helvetica", 16, "bold"))
        else:
            self.label_name.config(text=f"Predicted Name: {predicted_name}\nProbability: {round(prob,2)}", font=("Helvetica", 16, "bold"))

def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition App")
    parser.add_argument('--yolo_model', type=str, required=True, help="Path to YOLO model file")
    parser.add_argument('--facenet_model', type=str, required=True, help="Path to FaceNet model file")
    parser.add_argument('--svm_model', type=str, required=True, help="Path to SVM model file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    YOLO_model = YOLO(args.yolo_model)
    embedder = FaceNet()
    SVM_model = pickle.load(open(args.svm_model, 'rb'))
    
    root = tk.Tk()
    app = FaceRecognitionApp(root, YOLO_model, embedder, SVM_model)
    root.mainloop()