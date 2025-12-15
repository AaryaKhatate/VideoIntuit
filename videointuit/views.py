from django.shortcuts import render
from django.urls import reverse

def index(request):
    context = {
        'login_url': reverse('login'),
        # ... any other variables for your index.html ...
    }
    if request.user.is_authenticated:
        logout_url = reverse('logout')
        context['logout_url'] = logout_url
    return render(request, 'index.html', context)