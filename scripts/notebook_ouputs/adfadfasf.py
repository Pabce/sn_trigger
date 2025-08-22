import os

# print all folders in current directory (only the folders, not the files)
for folder in os.listdir():
    if os.path.isdir(folder):
        print(folder)
