from django.shortcuts import render,HttpResponse
from django.contrib import messages
from users.models import TitanicUserRegistrationModel
from .GetAlgorithms import AlgorithmsAccuracy
from users.models import TrainingModel
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import matplotlib


# matplotlib.use('nbagg')
matplotlib.use('TkAgg')
from django.conf import settings


# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('uname')
        pswd = request.POST.get('psw')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def UserList(request):
    data = TitanicUserRegistrationModel.objects.all()
    return render(request,'admins/UserList.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        TitanicUserRegistrationModel.objects.filter(id=id).update(status=status)
        data = TitanicUserRegistrationModel.objects.all()
        return render(request,'admins/UserList.html',{'data':data})


def AppliedAlgorithm(request):
    trainPath = settings.MEDIA_ROOT + "\\" + 'train.csv'
    testPath = settings.MEDIA_ROOT + "\\" + 'test.csv'

    obj = AlgorithmsAccuracy()
    rsltDict = obj.processAlgorithms(trainPath, testPath)
    print(rsltDict)

    return render(request, 'admins/Accuracy.html', {'rsltDict': rsltDict})


def AdminDataView(request):
    data_list = TrainingModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 20)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'admins/AdminViewData.html', {'users': users})

