# TransformerSummarization
Обучение и использование модели на основе архитектуры "трансформер" для генерации сумаризации текста.
 
Работа была выполнена в рамках домашнего задания курса Машинного обучения в МФТИ. Полная постановка задачи [тут](https://github.com/ml-dafe/ml_mipt_dafe/tree/main/05_Attention/homework)
Дополнительно накрутил всякие плюшки по типу wandb, dvc, kubernetes.. 

## Из интересного:
1. Есть возможность выбора эмбеддингов (дообучаются в процессе обучения) при инициализации модели 
   * ["fasttext"](https://fasttext.cc/docs/en/crawl-vectors.html)
   * ["bert"](https://huggingface.co/ai-forever/sbert_large_nlu_ru)
   * "None" - просто случайная матрица
2. Визуализация весов механизма attention после обучения модели на конкретном примере
3. Стандартная функция потерь, CrossEntropyLoss, заменена на LabelSmoothing. Хотим "сгладить" one-hot метки, чтобы модель не переобучалась.
4. В классический авторегрессионый метод генерации последовательности токенов добавлена "температура" для того, чтобы управлять "уверенностью" модели. 
5. Для оценки качества генерации текста используется ROUGE-2.
6. Добавлена возможность обучать модель трансформера не только параллельно, но и последовательно с применением teacher forcing подхода.
7. Обучение логгируется с использование wandb

   Предварительно необходимо выполнить
   > wandb login 
  
8. Код реализован в виде отдельных .py файлов и приложен dvc pipeline
    Спускаемся в папку dvc_pipeline и выполняем
    > dvc init --no-scm

    > dvc repro

9. Для тестирования работы модели в продакшн-подобном окружении был настроен локальный Kubernetes-кластер с помощью Minikube. Решение включает:
    + Docker-контейнер с REST API (на Flask), который:
      * Принимает текстовый запрос по HTTP/HTTPS.
      * Возвращает сгенерированную суммаризацию.
    + Minikube для локального развертывания:
      * Запуск пода (Pod) с моделью.
      * Доступ через Service (NodePort/LoadBalancer/Ingress).

    Для локального развертывания нужно: (Я делал из под Windows 10)
    Все команды из выполняются из корневой папки, в которой Dockerfile
    * Установить Docker, kubectl, minikube, не забыть вкл виртуализацию..
    * Запускаем кластер
       > ```
       > minikube start
       > ```
       
    * Собрать и залить образ в докер хаб
      В файле deploy.bat поменять "IMAGE_NAME=ilyaastapov/summary-prediction" на "IMAGE_NAME=<Ваш логин>/summary-prediction"
      > ```
      > .\delpoy.bat #выполняем (не быстрый процесс)
      > ```

    * Создать секрет для того, чтобы kubernetes мог скачать докер образ
       > ```
       > kubectl create secret docker-registry regcred --docker-server=https://index.docker.io/v1/ --docker-username=ваш_логин --docker-password=ваш_пароль --docker-email=ваш_email
       > ```
       
    * Запустить поды:
       >```
       > kubectl apply -f .\deployment.yaml # в этом .yaml нужно снова поменять "ilyaastapov" на ваш логин
       > kubectl apply -f .\service.yaml
       >```
     Должен быть такой статус

     ![image](https://github.com/user-attachments/assets/54b67075-e3f3-4026-9cc4-15175c010fdd)


    * Создать сетевой туннель между вашей локальной машиной и кластером Kubernetes

       >```
       > minikube tunnel # не закрывайте этот терминал
       >```
   Теперь можем обращаться к моделим например так через Postman:
 ![image](https://github.com/user-attachments/assets/072fc8be-3189-4f0c-b4e2-6ce2dad5dd49)
 ![image](https://github.com/user-attachments/assets/26d455cf-08ca-41b8-b692-9e5d5fa114ef)

    * Можем попробовать имитировать продакшн с помощью NGINX
       >```
       > # В новом терминале
       > minikube addons enable ingress
       > kubectl apply -f .\ingress.yaml
       > minikube service ingress-nginx-controller -n ingress-nginx
       > # В выводе  будет таблица, а под ней ingress-nginx ingress-nginx-controller и URL (он на и нужен)
       > # теперь запрос можем обращаться к динамическому порту
       >```
  ![image](https://github.com/user-attachments/assets/309e6c1c-0217-4da6-a48f-588b19e7cef7) 

   * 
