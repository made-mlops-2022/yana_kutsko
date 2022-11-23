### Build docker image:

<pre><code>docker build -t kutsko_mlops:1 .</code></pre>

### Pull docker image:

<pre><code>docker pull genusb/kutsko_mlops:2</code></pre>

### Run container:

<pre><code>docker run --rm -p 3000:3000  genusb/kutsko_mlops:2</code></pre>

## Steps to reduce docker image size:
1. Minimize number of layers. There are two COPY instructions: one for requirements.txt and one for the rest of the necessary files. It is split to two parts in order to use cache whenever possible
2. Install only required packages. For this requirements.txt was split for ml_project and for online_inference
3. Use light-weighted base image. Before: 1.46GB, after: 668MB