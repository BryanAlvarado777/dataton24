# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy your private SSH key and known hosts
# Make sure your private key has the correct permissions
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa

# Add GitHub to known_hosts to avoid SSH verification prompts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone the GitHub repository using SSH (replace YOUR-REPO with the actual repo)
RUN git clone git@github.com:BryanAlvarado777/dataton.git /app

# Install dependencies from the environment.yml file
RUN conda env create -f /app/environment.yml

# Make sure the environment is activated when running the container
SHELL ["conda", "run", "-n", "your-env-name", "/bin/bash", "-c"]

# Expose the port (if needed)
#EXPOSE 5000

# Start the application
#CMD ["python", "app.py"]
