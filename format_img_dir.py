import os
import itertools
import shutil

destination = "ADNI"

all_files = []
for root, _dirs, files in itertools.islice(os.walk(destination), 1, None):
    for filename in files:
        all_files.append(os.path.join(root, filename))

print(len(all_files))

for filename in all_files:
    shutil.move(filename, destination)