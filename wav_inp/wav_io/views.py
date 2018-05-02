from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from .forms import UploadFileForm
from .au_data1 import CreateDataAu
from .models import WavIO
import json

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            data_au = CreateDataAu()
            data_au.create_model(request.FILES['file'])
            return object(request, data_au.id)
    else:
        form = UploadFileForm()
    print(type(WavIO.objects.all()))
    return render(request, 'search.html', {'form': form, 'wav_list': WavIO.objects.all()[:10]})

def object(request, au_id):
    obj = get_object_or_404(WavIO, pk=au_id)
    data_au = CreateDataAu()
    data_au.create_recomendations(au_id=au_id)
    return render(request, 'object.html', {'wav_list': data_au.wav_list,
                                            'artist': obj.artist, 'albom': obj.albom,
                                            'title': obj.name, 'tag': json.loads(obj.ftag), 'id': au_id})
