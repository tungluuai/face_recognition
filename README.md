<!-- PROJECT LOGO -->
<br />
  <p align="center">
    <h1 style="text-align: center;">Face Recognition</h1>
  </p>
</div>

## Introduction
💡 The goal of this project is to develop an application that recognizes faces, with the potential for use in systems such as attendance tracking or authentication.

- This project proposes 2 approaches:
  
  🚀 **DenseNet121 model:** Train from scratch.  
  🚀 **YOLOv8 + FaceNet + SVM**: Face Alignment -> Embedding -> Classification.
  
- The training and validation sets are evaluated on four common metrics: Accuracy, Precision, Recall, F1-Score.
- Both approaches are capable of real-time deployment. Additionally, the second approach demonstrates efficiency in classifying multi-class and new classes with few samples (for personalization).

## Performance:
|                              | DenseNet121 | YOLOv8 + FaceNet + SVM |
|------------------------------|-------------|------------------------|
| **Training Set**                |             |                        |
| Accuracy                     | <div align="center">0.99</div>       | <div align="center">0.99</div>                   |
| Precision                    | <div align="center">0.99</div>       | <div align="center">0.99</div>                   |
| Recall                       | <div align="center">0.99</div>        | <div align="center">0.99</div>                   |
| F1 - Score                   | <div align="center">0.99</div>        | <div align="center">0.99</div>                   |
| **Validation Set**               |             |                        |
| Accuracy                     | <div align="center">0.83</div>        | <div align="center">**0.99**</div>                   |
| Precision                    | <div align="center">0.84</div>        | <div align="center">**0.99**</div>                   |
| Recall                       | <div align="center">0.83</div>        | <div align="center">**0.99**</div>                   |
| F1 - Score                   | <div align="center">0.83</div>        | <div align="center">**0.99**</div>                   |
| **Extra class**                  | <div align="center">❌</div>           | <div align="center">✅</div>                      |
| **Extra class with few samples** | <div align="center">❌</div>           | <div align="center">✅</div>                      |
| **Real-time Video**             | <div align="center">✅</div>           | <div align="center">✅</div>                     |
| **Inference time**              | <div align="center">Slower</div>      | <div align="center">**Faster**</div>                 |

## Demo of the second framework
- Test on extra class:
  
![Demo GIF 1](./demo/face.gif)
- Attendance system:
  
![Demo GIF 2](./demo/face_app.gif)
- Multiple class recognition:
  
![Demo GIF 3](./demo/multi.gif)
## Implementation
1. Clone the repo
    ```sh
    git clone https://github.com/tungluuai/face_recognition.git
    ```
2. Install packages:
    ```sh
    pip install -r requirements.txt
    ```
3. Download:
   + <a href="https://www.kaggle.com/datasets/troyct/105-classes-pins-dataset" target="_blank">Dataset</a>
   + <a href="https://github.com/ultralytics/ultralytics" target="_blank">Checkpoint</a> of YOLOv8.
   + <a href="https://github.com/a-m-k-18/Face-Recognition-System/blob/master/facenet_keras.h5" target="_blank">Checkpoint</a> of FaceNet.
5. If you want to:
   
    + Train and evaluate the **DenseNet121 model**, please follow the guideline in Jupyter notebook *DenseNet121.ipynb*.
    + Evaluate **the second framework**, please follow the guideline in Jupyter notebook *YOLOv8_FaceNet_SVM.ipynb*.
    + **Train extra datasets**, please follow the guideline in Jupyter notebook *Train_Extra_Data.ipynb*. Note that you must complete complete the procedure by step-by-step in *YOLOv8_FaceNet_SVM.ipynb* to find out hyperplane of SVM algorithm before transistion to performing this phase.
    + Refer to the attendance system:
    
    ```sh
    python app.py --yolo_model "../../load_model/yolov8n-face.pt" --facenet_model "../../load_model/facenet_model.pth" --svm_model "../../load_model/extra_SVM.pickle"
    ```

