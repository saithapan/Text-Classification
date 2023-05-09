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

        gstin_str = [i.strip() for i in img_text if 'gstin' in i.lower()]

        regex = r"GSTIN\s*:\s*(\w+)"

        final_resp = {}

        if gstin_str:
            total_gstin = []
            for each_gstin in gstin_str:
                resp = {}
                match_gstin = re.search(regex, each_gstin)
                if match_gstin:
                    gstin_num = match_gstin.group(1)
                    resp['GSTIN'] = gstin_num
                    total_gstin.append(resp)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            with open("output/" + timestamp + ".json", 'w+') as f:
                f.write(json.dumps(total_gstin))
            final_resp['body'] = total_gstin
        else:
            final_resp['body'] = "No GSTIN Number found"

        return Response(final_resp, status=status.HTTP_201_CREATED)
