from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel
# Create your views here.


def admin_login_check(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/admin_home.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/admin_home.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'admin_login.html', {})


def admin_home(request):
    return render(request, 'admins/admin_home.html')


def view_registered_users(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/view_registered_users.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/view_registered_users.html', {'data': data})


# Create your views here.
