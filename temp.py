# Hey

import shutil
import os

# List of directories to delete
dirs_to_delete = ['embeddings', 'models', 'checkpoint']
# dirs_to_delete = ['embeddings', 'models', 'checkpoint', 'dataset']

for dir_name in dirs_to_delete:
    dir_path = os.path.join(os.getcwd(), dir_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory not found: {dir_path}")