# 🛍️ Shoplifting Detection System (Django + Deep Learning)

This project is a **Django web application** that detects **shoplifting behavior in surveillance videos** using deep learning models (ConvLSTM or 3D-CNN).  
It allows you to upload video clips and receive predictions through a simple web interface.

---

## 📁 Project Structure

```
SHOP/
├── myproject/
│   ├── myproject/
│   │   ├── __init__.py
│   │   ├── asgi.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── predictor/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── apps.py
│   │   ├── models.py
│   │   ├── tests.py
│   │   ├── urls.py
│   │   ├── utils.py
│   │   ├── views.py
│   │   ├── best_shoplifting_model.h5 (Replaced with "shop-detection.ipynb"
│   │   └── templates/
│   │       └── index.html
│   ├── templates/
│   │   └── index.html
│   ├── db.sqlite3
│   └── manage.py
├── shop-detection.ipynb
└── README.md
```

---

## ⚙️ Features

- 📹 Upload surveillance video clips
- 🧠 Classifies videos as:
  - `shop lifters`
  - `non shop lifters`
- 📊 Real-time prediction results displayed on the web interface
- 💾 Model is pre-trained using ConvLSTM or 3D CNN architectures
- 🌐 Django backend to serve predictions

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/shoplifting-detection.git
   cd shoplifting-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**
   ```bash
   python manage.py migrate
   ```

5. **Run the Django server**
   ```bash
   python manage.py runserver
   ```

6. **Open in browser**
   ```
   http://127.0.0.1:8000/
   ```

---

## 🧠 Model Overview

### 🧩 ConvLSTM Model
- Pretrained `VGG16` (ImageNet) feature extractor
- `TimeDistributed` wrapper for per-frame features
- `LSTM` for temporal sequence learning
- Dense output layer for classification

### 📦 3D CNN Model
- `Conv3D` layers for spatio-temporal feature extraction
- `MaxPooling3D` and `BatchNormalization`
- `Dense` layers with `Dropout` for classification

> The trained `.h5` model is stored in `predictor/best_shoplifting_model.h5`

---

## 📁 Web Interface

- Located in `templates/index.html`
- Lets users upload video files
- Shows prediction results (shoplifter or not)

---

## 📋 Environment Requirements

- Python 3.8+
- Django 4.x
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib, scikit-learn

Install them manually or using:

```bash
pip install django tensorflow opencv-python scikit-learn matplotlib numpy
```

---

## 📊 Example Workflow

1. Place video datasets under:
   ```
   /Shop DataSet/shop lifters
   /Shop DataSet/non shop lifters
   ```
2. Train the model (`utils.py` / notebook version)
3. Save it as `best_shoplifting_model.h5`
4. Load model in `views.py` and serve predictions through Django

---

## 📄 License

MIT License — free to use, modify, and share.
