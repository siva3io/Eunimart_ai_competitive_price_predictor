#!/bin/bash
echo "Starting Nginx"
mkdir /run/nginx
nginx

echo "Starting python"
gunicorn --timeout 380 -w 2 --bind 127.0.0.1:9001 run:app