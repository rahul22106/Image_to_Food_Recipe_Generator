#!/bin/bash

sudo apt update 

sudo apt-get update 

sudo apt upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker $USER

newgrp docker

sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

sudo apt install git -y

sudo apt install awscli -y

git clone https://github.com/rahul22106/Image_to_Food_Recipe_Generator.git

cd Image_to_Food_Recipe_Generator

docker-compose up -d

docker-compose ps


## AWS configuration
aws configure


## Now setup elastic IP on AWS