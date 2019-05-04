import sys
import os

pattern = sys.argv[1]
print(pattern)

def get_result_line(i, pattern):
    filename = pattern.format(i)
    if os.path.exists(filename):
        lines = open(filename, 'r').readlines()[-3:]
        return str(i) + "\t" + '\t'.join([line.split(':')[1].strip() for line in lines])
    return None

results = []
for epoch in list(range(30)) + ['_best'] + ['_last']:
    result_line = get_result_line(epoch, pattern)
    results.append(result_line) if result_line is not None else None

print("epoch\tPrec\tRecall\tF_0.5")
for line in results:
    print(line)


