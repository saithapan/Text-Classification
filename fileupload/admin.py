from django.contrib import admin
from .models import FileData


# Register your models here.

class FileDataAdmin(admin.ModelAdmin):
    list_display = ('text', 'number')


admin.site.register(FileData, FileDataAdmin)
