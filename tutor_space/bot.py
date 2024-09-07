import docker

# Create a Docker client connected to the local Docker daemon
client = docker.from_env()

# Pull an image (e.g., the official Python image)
image = client.images.pull('python:3.9-slim')

# List all images
images = client.images.list()

# Print details of all images
for img in images:
    print(f"Image: {img.tags}")
