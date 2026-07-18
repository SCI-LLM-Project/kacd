import os

def read_txt_file_as_string (file_path): 
    with open(file_path, 'r') as file:
        file_content = file.read()
    
    return file_content


def load_files(variable):
    # Build the file paths for the annotations and descriptions directories
    annotations_path = os.path.join("variables_annotations", f"{variable}.txt")
    descriptions_path = os.path.join("variables_descriptions", f"{variable}.txt")

    # Read the content of each file
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_content = f.read()

    with open(descriptions_path, "r", encoding="utf-8") as f:
        descriptions_content = f.read()

    return annotations_content, descriptions_content



