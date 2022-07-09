FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "starter.py", "starter.py", "./"]

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT [ "python", "./starter.py" ]
