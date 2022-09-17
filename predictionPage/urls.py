from django.urls import path
from predictionPage import views

urlpatterns = [
    path('', views.indexPage, name="index"),
    path('result/', views.resultPage, name="result"),
    path('details/', views.detailPage, name="details"),
    path('news/', views.newsPage, name="news"),
    path('sentiment/news-<index>/', views.sentimentPage, name="sentiment"),
]