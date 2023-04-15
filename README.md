# servicepark_assignment

A Django-based API that allows users to upload text files containing a mixture of text and numeric data. The application processes the uploaded file asynchronously using celery and separates the text and numeric data into two separate columns, which are then saved in a SQLite database. 

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Setup](#setup)
4. [TestCases](#run-the-testcases)
5. [Run Project](#run-project)
6. [API Testing](#testing-the-api)
7. [Sample Response](#sample-response)

## Prerequisites

- Make sure Redis is install in the system already, if it's not present you can download it from
  here [Redis](https://github.com/tporadowski/redis/releases/download/v5.0.14.1/Redis-x64-5.0.14.1.msi)

## Installation

- Clone the code from this repo
- Create new environment and requirements.txt file

```commandline
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
```

## Setup

- To run the project initially we need to do migration
```shell
python .\manage.py makemigrations
python .\manage.py migrate
```

- Now open new terminal proceed for Running the celery, execute:
```shell
celery -A servicepack.celery worker --pool=solo -l info
```

## Run the testcases
- I written the two cases
   1. To check data is inserting in models table or not
   2. To check whether API is working or not
- To run the testcase, execute:-
```shell
python .\manage.py test
```

## Run Project

```shell
python .\manage.py runserver
```

## Testing the api

### Manually in Postman

- Once server started running
- Now Open Postman, select POST Method and provide this url(http://127.0.0.1:8000/upload)
- Now select Body -> form-data -> In key provide key name as `file` and select type as `file` and upload
  the `sample.txt` file and click send button
- ![image](https://user-images.githubusercontent.com/36238978/232121542-cf2f4e93-c6c2-4aa3-bd6b-03303eebff1b.png)

### By importing the collection

- In repo i provide the post collection file [here](https://github.com/saithapan/servicepack_assignment/blob/main/servicepark.postman_collection.json), you can import collection and test it directly 

## Sample Response
![image](https://user-images.githubusercontent.com/36238978/232181983-fea81054-d8d9-42a9-bb01-c9ae002551a2.png)
