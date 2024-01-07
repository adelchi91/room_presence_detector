systemctl start docker
docker build -t room_presence_image -f Dockerfile .
docker container ls
docker run -d --name container_room_presence -p 8000:8080 room_presence_image
docker container ls
docker container ls -a
curl -v 0.0.0.0:8000
docker container ls -a
docker tag room_presence_image adelchijacques/room_presence_classifier:room_presence_clf_tag
docker images
docker login
docker push adelchijacques/room_presence_classifier:room_presence_clf_tag
curl -v 0.0.0.0:8000
docker logs dcf13be0c4df
