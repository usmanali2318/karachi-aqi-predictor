import os, hopsworks

project = hopsworks.login(
    api_key_value=os.environ["HOPSWORKS_API_KEY"],
    project=os.environ["HOPSWORKS_PROJECT"]
)
print(f"Connected to project: {project.name}")