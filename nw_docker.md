## Run the `nw` Program with Docker Image

### Build and (Optionally) Publish the Image

On the local machine,

1.  Build the image
```
docker build -t neighborhoodwatch:latest .
```

2. Log in the docker-hub repository
```
docker login
```

3. Push the image to the repo
```
docker tag neighborhoodwatch:latest <docker_profile_id>/neighborhoodwatch:latest
docker push <docker_profile_id>/neighborhoodwatch:latest
```

### Prepare the Docker Running Environment 

On the target instance with GPU capablity (e.g. AWS EC2 `p3.8xlarge`), 

1. Install the docker engine ([doc](https://docs.docker.com/engine/install/))

2. Install `Nividia driver` (**NOTE** not CUDA toolkit) ([link](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#ubuntu-lts))

3. Install and configure `Nvidia docker container toolkit` ([link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation))

After the installation, make sure to configure the container runtime as below:
```
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker
```

### Run the Program within a Docker Container

On the target instance with GPU capablity,

1. Pull the docker image
```
$ docker pull <docker_profile_id>/neighborhoodwatch:latest
```

2. Start a docker container with the `nw` image 
```
$ docker run --rm --runtime=nvidia --gpus all --dit <docker_profile_id>/neighborhoodwatch
```

3. Log into the container 
```
$ NW_CONTAINER_ID=$(docker ps | grep neighborhoodwatch | awk '{print $1}')
$ docker exec -it ${NW_CONTAINER_ID} /bin/bash
```

4. Run the `nw` program inside the container
```
# poetry lock
# poetry install
# poetry run nw <...>
```

5. Last but not the least, exit the container and copy the generated datasets from the container to the target instance
```
$ docker cp ${NW_CONTAINER_ID}:/neighbourhoodwatch/knn_dataset/. .
```