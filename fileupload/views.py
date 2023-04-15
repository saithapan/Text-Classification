from .models import FileData
from .serializers import FileDataSerializer
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .tasks import process_text


class FileUploadView(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, format=None):

        data = request.data['file'].read().decode('utf-8')
        # Getting only text values
        total_text = ' '.join([each_word for each_word in data.split() if each_word.isalpha()])
        # getting only digits
        total_number = [int(each_word) for each_word in data.split() if each_word.isnumeric()]

        check_serializer_data = {'text': total_text, 'number': str(total_number)}
        serializer = FileDataSerializer(data=check_serializer_data)

        if serializer.is_valid():
            # record = serializer.save()
            # # For above created record will update the data in async
            process_text.delay(txt=total_text, nums=str(total_number))
            # checking the text contains these values or not
            words = ['hi', 'hello', 'bye', 'thanks', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
                     'nine', 'ten', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'and']
            is_words = {}
            resp = {}
            resp['text'] = total_text
            resp['numbers'] = total_number
            for word in words:
                if word in data:
                    is_words[word] = True
                else:
                    is_words[word] = False
                resp['is_words'] = is_words

            return Response(resp, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
