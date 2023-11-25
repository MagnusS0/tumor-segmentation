# Use the official Python image 
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the poetry files
COPY pyproject.toml poetry.lock /app/

# Install the specific version of poetry
RUN pip install poetry==1.7.1 && \
    poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-dev

# Copy the rest of the project
COPY . /app

# Set the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/app/src"


# Expose the port that the app runs on
EXPOSE 8501

# Execute the script when the container starts
CMD ["streamlit", "run", "src/app/app.py"]