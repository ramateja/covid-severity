FROM python:3.7
LABEL Maintainer ="Ram"

RUN mkdir -p /deploy/analytics
COPY ./requirements.txt /deploy/analytics
WORKDIR /deploy/analytics

VOLUME /deploy/analytics

RUN python -V
# RUN apt update
# RUN apt install libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip config --user set global.progress_bar on
VOLUME /root/.cache/pip
RUN pip install -r requirements.txt
COPY . .

ENTRYPOINT ["python"]
CMD ["run.py"]
