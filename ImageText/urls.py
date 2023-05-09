from django.urls import path

from .views import GetGstinFromImage

urlpatterns = [
    path('img_upload', GetGstinFromImage.as_view())
]
