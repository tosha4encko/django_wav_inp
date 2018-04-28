from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response, render
from .forms import UploadFileForm
from .au_data1 import CreateDataAu
from .models import WavIO
import os

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            data_au = CreateDataAu(request.FILES['file'])
            return render(request, 'index.html', {'form': form, 'wav_list': data_au.wav_list[:10],
                                                  'artist': data_au.artist, 'albom': data_au.albom, 'title': data_au.title})
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})

