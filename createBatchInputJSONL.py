import sys
from os import listdir
from os.path import isfile, join
import json

input_dir = sys.argv[1]
output_filename = sys.argv[2]

input_files = [join(input_dir, f) for f in listdir(input_dir) if isfile(join(input_dir, f))]
with open(output_filename, "w") as out_f:
    for input_file in input_files:
        with open(input_file, "r") as in_f:
            file_info = {"content": in_f.read()}
            out_f.write(f"{json.dumps(file_info)}\n")