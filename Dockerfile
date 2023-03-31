FROM ubuntu:18.04

RUN apt-get update

#Install required softwared for the module
RUN apt-get -y install python3.6 python3-pip nginx 


#Upgrade pip setup
RUN pip3 install --upgrade pip

#Install psycopg2 dependencies
RUN apt-get -y install libpq-dev
RUN pip3 install psycopg2 

# make a folder called 'src' where the application will reside
RUN mkdir /src

# sets project working directory to 'src'
WORKDIR /src

# copy app folder content to docker '/src/app' folder
COPY app /src/app
COPY config.py /src/config.py
COPY run.py /src/run.py
COPY constants.py /src/constants.py
COPY requirements.txt /src/requirements.txt
#COPY .env-sample /src/.env

# COPYING NGINX CONFIG FILE
COPY nginx.conf /src/nginx.conf
COPY nginx.conf /etc/nginx/sites-enabled/default
RUN chown www-data.www-data /src/nginx.conf

#Copying start script to run app
ADD start.sh /src/start.sh
RUN chmod +x /src/start.sh

# Install other thrid party packages required by the app
RUN pip3 install -r requirements.txt

EXPOSE 80
EXPOSE 9001

#Script to run nginx and gunicon
CMD ["/src/start.sh"]
