import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings
import os

# تحميل النموذج مرة واحدة عند بدء تشغيل التطبيق
MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'best_sholifting_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# إعدادات الفيديو (يجب أن تكون مطابقة لإعدادات التدريب)
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Shoplifting Detected", "No Shoplifting Detected"]

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

def predict_video(video_path):
    """
    الدالة الرئيسية التي تأخذ مسار الفيديو وتعيد التنبؤ.
    """
    # 1. استخلاص الإطارات
    frames = frames_extraction(video_path)

    # 2. التأكد من أن لدينا العدد الصحيح من الإطارات
    if len(frames) != SEQUENCE_LENGTH:
        return "Error: Could not extract the required number of frames."

    # 3. تحويل الإطارات إلى تنسيق مناسب للنموذج
    frames_np = np.asarray(frames, dtype=np.float32)
    # إضافة بُعد إضافي للـ batch (النموذج يتوقع دفعة من الفيديوهات)
    video_batch = np.expand_dims(frames_np, axis=0)

    # 4. عمل التنبؤ
    prediction = model.predict(video_batch)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASSES_LIST[predicted_class_index]

    return predicted_class_name