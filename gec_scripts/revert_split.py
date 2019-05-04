import sys

input_filename = sys.argv[1]
idx_filename = sys.argv[2]

lines = open(input_filename).readlines()
idxs = open(idx_filename).readlines()

new_lines = []
pre_idx = -1
for i, (line, idx) in enumerate(zip(lines, idxs)):
    line = line.strip()
    if pre_idx != idx:
        new_lines.append(line)
        pre_idx = idx
    else:
        new_lines[-1] = new_lines[-1] + line


for line in new_lines:
    print(line)

