import os

project_structure = [
    "templates",
    "static",
    "Notebook",
    "logs",
    "src",
    "src/components",
    "src/pipelines"
]

files = [
    ".gitignore",
    "app.py",
    "main.py",
    "README.md",
    "setup.py",

    # Notebooks
    "Notebook/data.txt",
    "Notebook/EDA.ipynb",
    "Notebook/model_trainer.ipynb",

    # Frontend
    "templates/index.html",
    "static/style.css",
    "static/script.js",

    # src core
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",

    # components
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",

    # pipelines
    "src/pipelines/__init__.py",
    "src/pipelines/prediction_pipeline.py",
    "src/pipelines/training_pipeline.py"
]

def create_folders():
    for folder in project_structure:
        os.makedirs(folder, exist_ok=True)
        print(f"Created Folder --> {folder}")

def create_files():
    for file in files:
        folder_path = os.path.dirname(file)

        if folder_path != "" and not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        if not os.path.exists(file):
            with open(file, "w") as f:
                pass
            print(f"Created File --> {file}")
        else:
            print(f"File Already Exists --> {file}")

if __name__ == "__main__":
    create_folders()
    create_files()
    print("\n✨ Project Structure Ready!")
