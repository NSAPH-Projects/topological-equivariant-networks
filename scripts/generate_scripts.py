import os

import yaml
from jinja2 import Environment, FileSystemLoader

# Define the path to your templates and experiments folders
templates_folder = "templates"
experiments_folder = "experiments"

# Create a Jinja2 environment
env = Environment(loader=FileSystemLoader(templates_folder))


# Function to read a YAML configuration file
def read_yaml_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


# Iterate over all templates in the templates folder
for template_file in os.listdir(templates_folder):
    if template_file.endswith(".j2"):
        # Load the template
        template = env.get_template(template_file)

        # Iterate over all experiment folders
        for experiment_name in os.listdir(experiments_folder):
            experiment_path = os.path.join(experiments_folder, experiment_name)
            if os.path.isdir(experiment_path):
                config_path = os.path.join(experiment_path, "config.yaml")
                if os.path.exists(config_path):
                    params = read_yaml_config(config_path)

                    # Render the script
                    script_content = template.render(params)

                    # Define the output script path
                    script_name = template_file.replace(".j2", "")
                    script_path = os.path.join(experiment_path, script_name)

                    # Write the generated script to a file
                    with open(script_path, "w") as script_file:
                        script_file.write(script_content)

                    print(f"Script {script_name} generated for {experiment_name} at {script_path}")

print("All scripts have been generated.")
