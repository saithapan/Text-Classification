from django.test import TestCase
from .models import FileData
from rest_framework import status


# Create your tests here.

class FileUploadTestCase(TestCase):

    def setUp(self) -> None:
        pass

    def test_models_insertion(self):
        """
        This test case will able to test whether inserting a record in db is working or not
        :return:
        """
        text = "this is a test text"
        number = str([100, 10, 30])
        obj = FileData.objects.create(
            text=text,
            number=number
        )
        self.assertEqual(text, obj.text)
        self.assertEqual(number, obj.number)

    def test_api_test(self):
        """
        This test case will call the api and check whether api is properly working or not
        :return:
        """
        with open('sample.txt', 'rb') as f:
            data = {'file': f}
            response = self.client.post("/upload", data, format='multipart')

        data = response.json()
        print(data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
