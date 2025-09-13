# predictor/views.py

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import predict_video
import os

def upload_predict(request):
    prediction_text = None
    video_url = None

    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        
        # حفظ الملف المؤقت
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        
        # الحصول على المسار الكامل للملف المحفوظ
        video_path = fs.path(filename)
        video_url = fs.url(filename) # مسار لعرض الفيديو في الصفحة
        
        # عمل التنبؤ
        prediction_text = predict_video(video_path)
        
        # (اختياري) حذف الفيديو بعد التنبؤ لتوفير المساحة
        # os.remove(video_path)

    return render(request, 'index.html', {'prediction': prediction_text, 'video_url': video_url})
