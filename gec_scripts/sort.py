import sys
import numpy as np
from collections import defaultdict

if len(sys.argv) != 3:
    print('Usage: <beam> <filename>')
    exit(-1)
beam = int(sys.argv[1])
filename = sys.argv[2]

x = sys.stdin.readlines()
hypo_dict = defaultdict(list)
for raw_line in x:
    raw_line_array = raw_line.strip().split('\t')
    hypo_dict[int(raw_line_array[0][2:])].append(raw_line_array[2])

ids = list(hypo_dict.keys())
line_lists = list(hypo_dict.values())

# sort line_lists by ids
idx = np.array(ids).argsort()

ofile = open(filename, 'w')
for line in np.array(line_lists)[idx]:
    ofile.write(line[0] + "\n")
ofile.close()
