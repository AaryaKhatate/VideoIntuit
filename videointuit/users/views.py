from django.shortcuts import render, redirect
from django.contrib.auth import login as auth_login, logout as auth_logout, authenticate
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm, LoginForm
from django.contrib import messages

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('index') # We'll define 'home' URL later
        else:
            messages.error(request, 'There was an error creating your account.')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def login(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            messages.success(request, f'Logged in as {user.username}!')
            return redirect('index') # We'll define 'home' URL later
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout(request):
    auth_logout(request)
    messages.info(request, 'Logged out successfully!')
    return redirect('index') # We'll define 'home' URL later

@login_required
def profile(request):
    return render(request, 'profile.html') # We'll create this template later
