import numpy as np
import sys

def split_line(line):
    result = []
    new_line = ""
    for i, c in enumerate(line):
        new_line += c

        if c == '.' and (i+1) < len(line) and str.isupper(line[i+1]):
            result.append(new_line)
            new_line = ""

    if len(new_line) > 0:
        result.append(new_line)

    return result

def split(filename, out_filename, index_filename):
    lines = open(filename).readlines()

    out_lines = []
    out_idxs = []
    for i, org_line in enumerate(lines):
        line = org_line.strip()
        sub_lines = split_line(line)
        for j, sub_line in enumerate(sub_lines):
            out_lines.append(sub_line.strip())
            out_idxs.append(i)


    with open(index_filename, 'w') as ofile:
        for idx in out_idxs:
            ofile.write(str(idx) + '\n')

    with open(out_filename, 'w') as ofile:
        for sl in out_lines:
            ofile.write(sl + '\n')

filename = sys.argv[1]
ofilename = sys.argv[2]
idx_filename = sys.argv[3]
split(filename, ofilename, idx_filename)
