from django.shortcuts import render
from django.urls import reverse

def index(request):
    context = {
        'login_url': reverse('login'),
        # ... any other variables for your index.html ...
    }
    return render(request, 'index.html', context)