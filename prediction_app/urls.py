from django.urls import path
from . import views

app_name = 'prediction_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_api, name='predict_api'),
    path('predict-batch/', views.batch_predict_api, name='batch_predict_api'),
]
