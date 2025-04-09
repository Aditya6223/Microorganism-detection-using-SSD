from django.shortcuts import render
from users.forms import UserRegistrationForm
# Create your views here.
def index(request):
    return render(request,'base.html')


# def index(request):
#     return render(request, 'index.html')


def register(request):
    form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


def admin_login(request):
    return render(request, 'admin_login.html', )


def user_login(request):
    return render(request, 'user_login.html', )

def doctor_login(request):
    return render(request,'doctor_login.html')
