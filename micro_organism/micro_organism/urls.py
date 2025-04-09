"""enterpriseman URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from micro_organism import views
from users import views as users_views
from admins import views as admins_views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('admin_login/', views.admin_login, name='admin_login'),
    path('user_login/', views.user_login, name='user_login'),
  

    # User side url
    path('user_register_action/', users_views.user_register_action,
         name='user_register_action'),
    path("user_login_check/", users_views.user_login_check, name="user_login_check"),
    path("user_home/", users_views.user_home, name="user_home"),
    path("upload_image/", users_views.upload_image, name="upload_image"),
    path("detect_objects/", users_views.detect_objects, name="detect_objects"),
 

    # Admins side url
    path('admin_login_check/', admins_views.admin_login_check,
         name='admin_login_check'),

    path('admin_home/', admins_views.admin_home, name='admin_home'),

    path('view_registered_users/', admins_views.view_registered_users,
         name='view_registered_users'),

    path("AdminActivaUsers/", admins_views.AdminActivaUsers,
         name="AdminActivaUsers"),
    
    
    
    

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)