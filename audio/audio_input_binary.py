import struct
import array

# open函数以二进制模式打开文件，指定mode参数为'b'
# 二进制数据可以用readinto读入到提取分配好的buffer中，便于数据处理
# 解析二进制数据可以使用标准库中的struct模块的unpack方法

f = open('.wav','rb')
# 读取文件前44个字节字符
info = f.read(44)

# 解析二进制数据-- 声道数
print(struct.unpack('h',info[22:24]))

# WAV文件中data部分的大小/数据类型的大小->buffer大小
f.seek(0,2)
n = (f.tell() - 44) / 2

buf = array.array('h',(0,for _ in  range(n)))

# 将数据读至buf中
f.seek(44)
f.readinto(buf)

for i in range(n):
    # 数据处理，相当于将音频文件的声音变小
    buf[i] /= 8

f.close()
