FROM ubuntu:20.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" , "app.py" ]
