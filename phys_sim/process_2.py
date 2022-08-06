import numpy as np
import codecs
def read_data(data):
  new = data.split('\n')
  B = []
  C = []
  for j in range(256):
    for i in range(len(new[j].split(' '))):
      if i%2 != 0:
        B.append(float(new[j].split(' ')[i]))
  for j in range(257, 513):
    for i in range(len(new[j].split(' '))):
      if i%2 != 0:
        C.append(float(new[j].split(' ')[i]))
  print(np.asarray(B).shape)
  B_f = np.asarray(B).reshape((256, 256))
  C_f = np.asarray(C).reshape((256, 256))  
  return [B_f, C_f]

temp_B = []
temp_C = []
for i in range(2):
  for j in range(500,600,1):
    if j<10:
      f = codecs.open(u'./bintemp/'+f'{i}'+'prof_gp.00000'+f'{j}','r','utf-8')
    if 10<=j and j< 100:
      f = codecs.open(u'./bintemp/'+f'{i}'+'prof_gp.0000'+f'{j}','r','utf-8')
    if 100<=j<1000:
      f = codecs.open(u'./bintemp/'+f'{i}'+'prof_gp.000'+f'{j}','r','utf-8')
    if 1000<=j<10000:
      f = codecs.open(u'./bintemp/'+f'{i}'+'prof_gp.00'+f'{j}','r','utf-8')
    if 10000<=j<100000:
      f = codecs.open('./bintemp/'+f'{i}'+'prof_gp.0'+f'{j}', 'r', 'utf-8')
    data = f.read()
    f.close()
    B, C = read_data(data)

    temp_B.append(B)
    temp_C.append(C)
  print(i)  
comp_B = np.reshape(np.array(temp_B), (2, 100, 256, 256))
comp_C = np.reshape(np.array(temp_C), (2, 100, 256, 256))

comp_B = np.transpose(comp_B, (0,2,3,1))
comp_C = np.transpose(comp_C, (0,2,3,1))

Data = np.array([comp_B, comp_C])
Data = np.transpose(Data, (1,2,3,0,4))

np.save('Data_fno3_res256_dt1.npy', Data)


