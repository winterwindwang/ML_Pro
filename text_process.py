import os
import sys
import io

filename = r'D:\student\英语\英语.txt'
str=[]
with open(filename,'r') as fr:
    lines = fr.readlines()
    for line in lines:
        str.append(line)

# with open(filename,'w') as fr:
#     fr.write(str)
print(str)
print('success!')