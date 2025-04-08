from django.urls import path
from . import views

urlpatterns = [
    path('upload_video/', views.upload_video, name='upload_video'),
    path('ask_question/', views.ask_question, name='ask_question')
    # REMOVE this line: path('', views.index, name='index'),
]