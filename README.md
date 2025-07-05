# 🛡️ WorkSafe AI – PPE Detection System with YOLOv8 and Flask

This is a deep learning-powered web application for real-time **Personal Protective Equipment (PPE)** detection. It uses YOLOv8 to detect safety gear like helmets, vests, gloves, etc., and provides safety scores via a Flask-based web interface.

Developed by **Sümeyye İclal Ezber**, **Sümeyye Dilmaç**, and **Çağla Kaçar** as part of a collaborative graduation project.



## 🚀 Features

- 🎥 Real-time detection via webcam, video, or image
- 🧠 YOLOv8-based object detection (17 PPE classes)
- 📊 Automatic safety score calculation (Low / Medium / High)
- 📈 Interactive charts and class distributions (via Chart.js)
- 📄 Export detection logs to Excel and PDF
- 💻 Clean and responsive Flask-based web UI


## 📂 Dataset

Model was trained on the [SH17 PPE Detection Dataset (Kaggle)](https://www.kaggle.com/datasets), including the following 17 classes:

['person', 'head', 'face', 'glasses', 'face-mask-medical', 'face-guard',
'ear', 'ear-muffs', 'hands', 'gloves', 'foot', 'shoes',
'safety-vest', 'tools', 'helmet', 'medical-suit', 'safety-suit']




## 🧰 Technologies Used

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- Flask (Python Web Framework)
- OpenCV for image/video processing
- Chart.js for data visualization
- Pandas & fpdf2 for logging and export
- HTML/CSS/JS for frontend



## 📦 Installation:

``bash
git clone https://github.com/iclalezber/WorkSafe-AI-Smart-PPE-Detection-System.git
cd WorkSafe-AI-Smart-PPE-Detection-System
pip install -r requirements.txt

⚠️ Don’t forget to place your trained YOLOv8 model (best.pt) into the /models/ folder.



## ▶️ Run the App:
python app.py
Then open your browser and go to:
http://127.0.0.1:5000



## 📊 Output Examples
*All detected images and videos are saved in static/detected/

*Detection statistics and charts are shown on the /chart page

*Logs can be downloaded from the /log page as Excel or PDF



## 📜 License
This project is licensed under the MIT License.
