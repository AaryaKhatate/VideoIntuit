# videointuit/urls.py
from django.contrib import admin
from django.urls import include, path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Ensure this line exists
    path('upload_video/', views.upload_video, name='upload_video'),
    path('ask_question/', views.ask_question, name='ask_question')
]
