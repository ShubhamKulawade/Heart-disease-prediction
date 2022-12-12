from django.urls import path
from . import views
urlpatterns=[
	path('',views.home,name='home'),
	path('about',views.about,name='about'),
	path('contact',views.contact,name='contact'),
	path('result',views.result,name='result'),
	path('login',views.login,name='login'),
	path('register',views.register,name='register'),
	path('success',views.success,name='success'),
	path('prediction',views.predict,name='prediction'),
	path('eda',views.eda,name='eda'),
	path('dp',views.prediction,name='dp'),
	path('simple_upload',views.simple_upload,name='simple_upload'),
	path('advice',views.advice,name='advice'),
]