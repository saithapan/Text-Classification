# servicepark_assignment

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

- To run the project initially we need run the celery, execute:

```shell
celery -A servicepack.celery worker --pool=solo -l info
```

- Now in new terminal proceed for migration

```shell
python .\manage.py makemigrations
python .\manage.py migrate
```

- Run Project

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

- In repo i provide the post collection file, you can import collection and test it directly 
