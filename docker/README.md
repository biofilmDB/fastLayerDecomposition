## Easy installation

You don't need to clone or download this, since the Dockerfile has been pushed to Docker Hub. You can simply install via:

    docker pull cragl/fastlayers

and run the server via:

    docker run -p 8000:8000 -p 9988:9988 cragl/fastlayers

Open your web browser to: http://localhost:8000/

This Docker installation doesn't use OpenCL.

## Without DockerHub

If you'd like to build without docker hub, you can run the following without checking out this repository:

    docker build -f docker/Dockerfile -t fastlayer https://github.com/CraGL/fastLayerDecomposition.git

If you want to edit the code, check out the repository, `cd fastLayerDecomposition`, and then:

    docker build -f docker/Dockerfile -t fastlayers .

Run the server via:

    docker run -p 8000:8000 -p 9988:9988 fastlayers
