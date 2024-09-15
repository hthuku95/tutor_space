import docker

# Create a Docker client connected to the local Docker daemon
client = docker.from_env()

# image = client.images.pull('python:3.9-slim')

images = client.images.list()

for img in images:
    print(f"Image: {img.size}")
