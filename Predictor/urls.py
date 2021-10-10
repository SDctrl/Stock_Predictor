
from django.urls import path 
from django.urls.resolvers import URLPattern
from . import views

Urlpattens = [
    path('', views.load_data, name="load_data") ,
    path('', views.plot_raw_data, name="plot_raw_data") ,
    path('', views.Get_or_not, name="Get_or_not" ),

]


