from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def home(request):
    return render(request,'home.html')

def aboutUs(request):
    return HttpResponse("Welcome to My Homepage")
def Course(request):
    return HttpResponse("Welcome to My Homepage")
def courseDetails(request,courseid):
    return HttpResponse(courseid)
def predict(request):
    return render(request,'predict.html')
def result(request):
    diabetes_dataset = pd.read_csv(r'C:\Users\rupok\Downloads\diabetes.csv')
    X= diabetes_dataset.drop(columns='Outcome',axis=1)
    Y=diabetes_dataset['Outcome']
    scaler=StandardScaler()
    scaler.fit(X)
    standarized_data=scaler.transform(X)
    X=standarized_data
    Y=diabetes_dataset['Outcome']
    

    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    val5=float(request.GET['n5'])
    val6=float(request.GET['n6'])
    val7=float(request.GET['n7'])
    val8=float(request.GET['n8'])

    input_data = scaler.transform([[val1, val2, val3, val4, val5, val6, val7, val8]])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    pred = model.predict(input_data)
    result1 = "Positive" if pred[0] == 1 else "Negative"
    return render(request, "predict.html", {"result2": result1})