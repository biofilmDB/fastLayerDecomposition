You don't need to clone or download this, since the Dockerfile has been pushed to Docker Hub. You can simply install via:

    docker pull cragl/fastlayers

and run the server via:

    docker run -p 8000:8000 -p 9988:9988 cragl/fastlayers

Open your web browser to: http://localhost:8000/

This Docker installation doesn't use OpenCL.

If you'd like to build without docker hub, you can run the following next to the Dockerfile:

    docker build -t fastlayers .

Run the server via:

    docker run -p 8000:8000 -p 9988:9988 fastlayers

By default, this Dockerfile downloads the code from GitHub.
That can make editing the code a pain, since you have to push to GitHub each time you want to run it. Instead, you can move this Dockerfile to the root of this repository and change the "Download the code" option in the Dockerfile to copy the code locally.
