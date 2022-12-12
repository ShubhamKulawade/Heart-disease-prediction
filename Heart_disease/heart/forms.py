from django import forms
from.models import *



class con(forms.ModelForm):
	class Meta:
		model=Contact
		fields={'name','phone','email','msg'}


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()



class upload(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()