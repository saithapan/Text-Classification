# cognida_assignment

## Prerequisites

- Make sure tesseract  is install in the system already, if it's not present you can download it from
  here [tesseract](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe)

## Installation

- Clone the code from this repo
- Create new environment and requirements.txt file

```commandline
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```

## Setup

- To run the project initially we need assign the application path of tesseract in config.py file:

```shell
config = {
    "tesseract": r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with you tesseract path
}
```

- Run Project in New terminal

```shell
python .\manage.py runserver
```
## Django App main files explanation
1. `ImageText/urls.py` - Main app script containing the API endpoints 
2. `ImageText/views.py` - Actual logic written here in class based views 
3. `config.py` - App configurations details like tesseract path

## Testing the api

### In Postman

- Once server started running
- Now Open Postman, select POST Method and provide this url(http://127.0.0.1:8000/img_upload)
- Now select Body -> form-data -> In key provide key name as `file` and select type as `file` and upload
  the `sample.png` file and click send button
- ![image](https://github.com/saithapan/cognida_assignment/assets/36238978/44b1993b-7662-48ad-a1bf-cb132acd2383)
- **Output Sample Image:-**
- ![image](https://github.com/saithapan/cognida_assignment/assets/36238978/65db4ddf-b880-4374-86d3-cf456c17c955)
- **If it has multiple GSTIN Numbers(sample img in input/3.jpg) in Invoice then output will be :-**
- ![image](https://github.com/saithapan/cognida_assignment/assets/36238978/54787d0a-3968-4ffd-bee0-5875d314129b)

