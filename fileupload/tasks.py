from celery import shared_task
from .models import FileData


@shared_task
def process_text(txt, nums):
    print("in tasks")
    file_data = FileData(text=txt, number=nums)
    file_data.save()
