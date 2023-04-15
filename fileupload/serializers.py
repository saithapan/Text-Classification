from rest_framework import serializers
from .models import FileData


class FileDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileData
        fields = "__all__"

    # def create(self, validated_data):
    #     return FileData(**validated_data)
