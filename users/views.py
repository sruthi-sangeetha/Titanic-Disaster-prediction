from django.shortcuts import render, HttpResponse
from .forms import TitanicUserRegistrationForm
from .models import TitanicUserRegistrationModel
from django.contrib import messages
import io, csv
from django.conf import settings
# Create your views here.
import pandas as pd
from .models import TrainingModel, TestingModel
from .DataPreprocess import ProcessData
from django_pandas.io import read_frame
from .AllAlgorithmsScores import Algorithms
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import matplotlib

# matplotlib.use('nbagg')
matplotlib.use('TkAgg')


def UserRegisterAction(request):
    if request.method == 'POST':
        form = TitanicUserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = TitanicUserRegistrationForm()
            return render(request, 'UserRegister.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = TitanicUserRegistrationForm()
    return render(request, 'UserRegister.html', {'form': form})


def UserLoginCheckAction(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = TitanicUserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UploadDataForm(request):
    return render(request, 'users/UploadDataForm.html', {})


def UploadCSVToDataBase(request):
    filepath = settings.MEDIA_ROOT + "\\" + 'train.csv'
    data = pd.read_csv(filepath)
    data = data.fillna(0)
    # print('Data',data.head(10))
    for i in data:
        print(data['Name'])
        _, created = TrainingModel.objects.update_or_create(
            PassengerId=data['PassengerId'],
            Survived=data['Survived'],
            Pclass=data['Pclass'],
            Name=data['Name'],
            Sex=data['Sex'],
            Age=data['Age'],
            SibSp=data['SibSp'],
            Parch=data['Parch'],
            Ticket=data['Ticket'],
            Fare=data['Fare'],
            Cabin=data['Cabin'],
            Embarked=data['Embarked']

        )

    # declaring template
    template = "users/UserHome.html"
    data = TrainingModel.objects.all()
    # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address,    phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    # if request.method == "GET":
    #     return render(request, template, prompt)
    # csv_file = request.FILES['file']
    # # let's check if it is a csv file
    # if not csv_file.name.endswith('.csv'):
    #     messages.error(request, 'THIS IS NOT A CSV FILE')
    # data_set = csv_file.read().decode('UTF-8')
    #
    # # setup a stream which is when we loop through each line we are able to handle a data in a stream
    # io_string = io.StringIO(data_set)
    # next(io_string)
    # for column in csv.reader(io_string, delimiter=',', quotechar="|"):
    #     print('column ',column)
    #     #print('Data Set ',column[0],column[1],column[2],column[3],'Sex =',column[4],'Age = ',column[5])
    #     _, created = TrainingModel.objects.update_or_create(
    #         PassengerId=column[0],
    #         Survived=column[1],
    #         Pclass=column[2],
    #         Name=column[3]+','+column[4],
    #         Sex=column[5],
    #         Age=column[6],
    #         SibSp=column[7],
    #         Parch=column[8],
    #         Ticket=column[9],
    #         Fare=column[10],
    #         Cabin=column[11],
    #         Embarked=column[12],
    #
    #     )
    # context = {}

    return render(request, 'users/UserHome.html', {})


def DataPreProcess(request):
    trainPath = settings.MEDIA_ROOT + "\\" + 'train.csv'
    testPath = settings.MEDIA_ROOT + "\\" + 'test.csv'
    train = TrainingModel.objects.filter().order_by('id')[:10]
    test = TestingModel.objects.all()
    # trainPath = read_frame(train)
    # testPath = read_frame(test)
    obj = ProcessData()
    obj.process(trainPath, testPath)

    return render(request, 'users/CleanedData.html', {'data': train})
    # return HttpResponse(html)


def AlgorithmResult(request):
    trainPath = settings.MEDIA_ROOT + "\\" + 'train.csv'
    testPath = settings.MEDIA_ROOT + "\\" + 'test.csv'

    obj = Algorithms()
    rsltDict = obj.processAlgorithms(trainPath, testPath)
    print(rsltDict)

    return render(request, 'users/AlgorithmAccuracy.html', {'rsltDict': rsltDict})


def TestUser(request):
    return render(request, 'users/FindSurvival.html', {})


def SearchSurvival(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        try:
            check = TrainingModel.objects.filter(Name__contains=name)
            for x in check:
                survival = x.Survived
                if survival>=1:
                    messages.success(request, 'The Persons ' + x.Name + ' has survived')
                    #return render(request, 'users/FindSurvival.html', {})
                else:
                    print("Home not Run = ", survival)
                    messages.success(request, 'The Persons ' + x.Name + ' not has survived')
                    #return render(request, 'users/FindSurvival.html', {})


        except Exception as ex:
            messages.success(request, 'Data not found')
            pass

    return render(request, 'users/FindSurvival.html', {})

def TotalSurvival(request):
    check = TrainingModel.objects.filter(Survived=1)
    page = request.GET.get('page', 1)

    paginator = Paginator(check, 20)
    try:
        check = paginator.page(page)
    except PageNotAnInteger:
        check = paginator.page(1)
    except EmptyPage:
        check = paginator.page(paginator.num_pages)
    return render(request,'users/TotalSurvivals.html',{'data':check})