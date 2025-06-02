"""ExploratoryTitanic URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ExploratoryTitanic import views as mainView
from users import views as usr
from admins import views as admins
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', mainView.index, name='index'),
    path('logout/', mainView.logout, name='logout'),
    path('UserLogin/', mainView.UserLogin, name='UserLogin'),
    path('AdminLogin/', mainView.AdminLogin, name='AdminLogin'),
    path('UserRegister/', mainView.UserRegister, name='UserRegister'),

    ##--> User Side Urls <--##
    path('UserRegisterAction/',usr.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheckAction/', usr.UserLoginCheckAction, name='UserLoginCheckAction'),
    path('UploadDataForm/', usr.UploadDataForm, name='UploadDataForm'),
    path('UploadCSVToDataBase/', usr.UploadCSVToDataBase, name='UploadCSVToDataBase'),
    path('DataPreProcess/', usr.DataPreProcess, name='DataPreProcess'),
    path('AlgorithmResult/', usr.AlgorithmResult, name='AlgorithmResult'),
    path('TestUser/', usr.TestUser, name='TestUser'),
    path('SearchSurvival/', usr.SearchSurvival, name='SearchSurvival'),
    path('TotalSurvival/', usr.TotalSurvival, name='TotalSurvival'),




    ##--> Admin Side Urls <--##
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('UserList/', admins.UserList, name='UserList'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),
    path('AppliedAlgorithm/', admins.AppliedAlgorithm, name='AppliedAlgorithm'),
    path('AdminDataView/', admins.AdminDataView, name='AdminDataView'),


]
