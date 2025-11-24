# 🚗 Vehicle Detection & Counting using YOLOv8 + Streamlit

This project detects and counts vehicles (cars, buses, motorcycles, trucks) from a traffic video using **YOLOv8** and displays real-time analytics on a **Streamlit dashboard**.

It shows:
- Live processed video with bounding boxes  
- Vehicle counting using a crossing line  
- Real-time bar chart of vehicle count  
- Real-time line chart of total vehicles  

---

## 🧠 Features
- YOLOv8 object detection  
- Multi-object tracking  
- Accurate vehicle counting using centroid tracking  
- Streamlit dashboard UI  
- Real-time visual analytics (charts + video)  
- Configurable upload size  

---


## Run locally
1. Create venv: `python -m venv venv`
2. Activate venv (Windows): `venv\Scripts\activate`
3. Install: `pip install -r requirements.txt`
4. Run: `streamlit run src/main.py`

