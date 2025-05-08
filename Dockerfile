FROM arm64v8/python:3.11

RUN apt-get update
RUN apt-get install -y sudo locales apt-utils tzdata init systemd
RUN apt-get install -y locales && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
RUN apt-get install -y vim less

ENV TIMEZONE=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TIMEZONE /etc/localtime \
    && echo $TIMEZONE > /etc/timezone 

#ENV LANG ja_JP.UTF-8
#ENV LANGUAGE ja_JP:ja
#ENV LC_ALL_ja_JP.UTF-8
#ENV TZ JST-9
#ENV TERM xterm

RUN pip3 install --upgrade setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN /usr/bin/systemctl disable getty@tty1.service \
  && /usr/bin/systemctl disable getty.target \
  && /usr/bin/systemctl disable systemd-udevd.service

