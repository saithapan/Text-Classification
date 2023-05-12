from PIL import Image
import re
import pytesseract
import json
from config import config
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status
import datetime

# loading the tesseract application from config.py
pytesseract.pytesseract.tesseract_cmd = config['tesseract']


class GetGstinFromImage(APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, format=None):

        data = request.data['file']
        img_text = pytesseract.image_to_string(Image.open(data))
        img_text = img_text.split('\n')

        pairs = {}
        for i in range(len(img_text)):
            e = img_text[i]
            e_split = e.split()
            for num, each_word in enumerate(e_split):
                try:
                    if ':' == each_word:
                        pairs[e_split[num - 1]] = e_split[num + 1]
                    elif ':' == each_word[-1]:
                        try:
                            pairs[e_split[num]] = e_split[num+1]
                        except:
                            print("not found")
                except:
                    print("error")

        final_resp = {}


        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if pairs:
            with open("output/" + timestamp + ".json", 'w+') as f:
                f.write(json.dumps(pairs))
            final_resp['body'] = pairs
        else:
            final_resp['body'] = "No GSTIN Number found"

        return Response(final_resp, status=status.HTTP_201_CREATED)
