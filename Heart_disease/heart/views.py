from django.shortcuts import render, redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from . models import *
from . forms import *
from django.http import HttpResponse

#data Science packages
import numpy as np 
import pandas as pd 
#Import library about data visualization 
import seaborn as sns
import matplotlib.pyplot as plt

#Machine learning packages

from sklearn.preprocessing import StandardScaler
# from .models import Document
from .forms import UploadFileForm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Create your views here.
def home(request):
	return render(request,'index.html')
def contact(request):
	c=con(request.POST or None)
	if c.is_valid():
		c.save()
		return redirect('/')
	return render(request,'contact.html',{'x':c})
	
def about(request):
	return render(request,'about.html')
def advice(request):
	return render(request,'advice.html')
def predict(request):
	return render(request,'user_predict.html')	
def register(request):
	if request.method=='POST':
		fn=request.POST['name']
		em=request.POST['email']
		un=request.POST['username']
		psw1=request.POST['psw']
		psw2=request.POST['psw1']
		if psw1==psw2:
			if User.objects.filter(username=un).exists():
				messages.warning(request,'username already exist try again')
				return redirect('register')
			else:
				usr=User.objects.create_user(first_name=fn,email=em,username=un,password=psw1)
				usr.save()
				return redirect('login')
		else:
			messages.warning(request,'not matching password')		
	return render(request,'register.html')
def login(request):
	if request.method=='POST':
		u=request.POST['username']
		p=request.POST['password']
		ur=auth.authenticate(username=u,password=p)
		if ur is not None:
			auth.login(request,ur)
			return redirect('prediction')
		else:
			return redirect('login')
	return render(request,'login.html')
def success(request):
    return HttpResponse('successfully uploaded, Thank you , we will contact you')


# custom method for generating predictions
def getPredictions(age,sex,	cp, trestbps,chol,fbs,	restecg,thalach,exang,oldpeak,slope,ca,thal):
    import pickle
    model = pickle.load(open("C:/Users/Shubham/Desktop/TechCiti/heart/heart.sav", "rb"))
    scaled = pickle.load(open("C:/Users/Shubham/Desktop/TechCiti/heart/scaler.sav", "rb"))
    prediction = model.predict(scaled.transform([[age,sex,	cp, trestbps,chol,fbs,	restecg,thalach,exang,oldpeak,slope,ca,thal]]))
    
    if prediction == 0:
        return "Your heart is safe"
    elif prediction == 1:
        return "Sorry you have heart disease"
    
    else:
        return "error"	

# our result page view
def result(request):
	age = int(request.GET['age1'])
	sex = int(request.GET['sex'])
	cp = int(request.GET['cp'])
	trestbps = float(request.GET['trestbps'])
	chol = float(request.GET['chol'])
	fbs = float(request.GET['fbs'])
	restecg = float(request.GET['restecg'])
	thalach = float(request.GET['thalach'])
	exang= float(request.GET['exang'])
	oldpeak=float(request.GET['oldpeak'])
	slope=float(request.GET['slope'])
	ca=float(request.GET['ca'])
	thal=float(request.GET['thal'])
	result = getPredictions(age,sex,cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)

	return render(request, 'result.html', {'result':result})

def eda(request):

	data=pd.read_csv("E:/atees/Project2020/Heart Disease Prediction/Final june/heartprediction/predictionform/heart.csv")
#Get information about data
	data.info()
	#To seen first 10 rows 
	data.head(10)
	#seen data correlation in heatmap 
	f,ax=plt.subplots(figsize=(18,18))
	sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.2f',ax=ax)
	data.corr()

	print(data.corr())
	# Line Plot"trestbps(The person's resting blood pressure) and thalach (The person's maximum heart rate achieved)" attributes
	#Note: Attributes have negative correlation that only using only show plot 
	data.trestbps.plot(kind="line",color="g",label="age",linewidth=1,grid=True,linestyle=":")
	data.thalach.plot(color="r",label="chol",linewidth=1,grid=True,linestyle="-")

	 
	plt.title("Line Plot")
	plt.xlabel=('x axis')
	plt.ylabel=('y axis')
	# Scatter Plot"trestbps(The person's resting blood pressure) and thalach (The person's maximum heart rate achieved)" attributes 
	data.plot(kind='scatter',x='trestbps',y='thalach',color='blue')
	plt.title('Scatter Plot')
	#Hstogram 
	data.trestbps.plot(kind="hist",bins=50,figsize=(15,15))
	plt.title("Histogram Plot")
	#Data Filtering Logical 
	data[(data["trestbps"]>130)&(data["chol"]>210)]
	#Data Filtering Logical
	data_logicfilter=data[np.logical_and(data["age"]>60,data["trestbps"]>170)]
	data_logicfilter
	#Getting data with using loops
	for index,value in data[["trestbps"]][0:20].iterrows():
	    print(index,":",value)

	    #Using Lambda function and calculated mean of age 
	square = lambda x,y: x/y    
	print(square(data["age"].sum(),len(data["age"])))
	#Built in function mean()
	print(data["age"].mean())
	# iterating cholestrol data
	data_iter =data["chol"]
	it = iter(data_iter)
	print(next(it))    #print first row  next iteration
	print(it)         # print other rows remaining iteration
	#Using List Comprehension
	threshold = data["chol"].sum()/len(data["chol"])
	print(threshold,data["chol"].mean())
	data["chol_threshold"]=["High Chol"if i>threshold else "Low Chol" for i in data["chol"]]
	data.loc[:20,["chol_threshold","chol"]] 
	#rearrange columns so that target column is last.
	data=data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','chol_threshold','target']]
	data.head()
	# A Box plot is a hybrid of a box plot and a kernel density plot, which shows peaks in the data.
	cols = data.columns
	size = len(cols) - 1 # We don't need the target attribute
	# x-axis has target attributes to distinguish between classes
	x = cols[size]
	y = cols[0:size]

	for i in range(0, size):
	    sns.boxplot(data=data, x=x, y=y[i])
	    plt.show()
	plt.show()
	return render(request,'index.html')
def simple_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            instance = ModelWithFileField(file_field=request.FILES['file'])
            instance.save()
            return HttpResponseRedirect('index')
    else:
        form = UploadFileForm()
    return render(request, 'doctor_predict.html', {'form': form})	

def prediction(request):
	dataset = pd.read_csv('E:/atees/Project2020/Heart Disease Prediction/Final june/heartprediction/predictionform/heart.csv')
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	
	classifier = SVC(kernel = 'linear', random_state = 0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
	
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	b=accuracy_score(y_test, y_pred)
	print(b)
	return render(request,'result1.html',{'a':cm,'b':b})
