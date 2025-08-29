from collections import defaultdict

mydict = defaultdict(list)

for i in range(10):
    mydict[i].append(i+10)

print(mydict.values())