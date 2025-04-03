@echo off
set IMAGE_NAME=ilyaastapov/summary-prediction
set TAG=latest

echo Собираем Docker-образ...
docker build -t %IMAGE_NAME%:%TAG% .

echo Отправляем образ в Docker Hub...
docker push %IMAGE_NAME%:%TAG%

echo Готово!
pause
