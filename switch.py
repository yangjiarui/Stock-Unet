# encoding: utf-8
'''
@author: headsome jerry
@time:18-5-27下午9:05
@file_name:switch.py
'''
from current_unet import Main,evaluate_mode

all={}
for i in range(0,15,3):
    resualt_file=Main(start=i,end=i+3,interval=1)
    #acc1=evaluate_mode(resualt_file)
    all[str(i)+' to '+str(i+1)+"'s acc: "]=evaluate_mode(resualt_file)
    print(str(i)+'Done')


# resualt_file=Main(start=20,end=25,interval=1)
# acc2=evaluate_mode(resualt_file)
# resualt_file=Main(start=20,end=50,interval=1)
# acc3=evaluate_mode(resualt_file)
# resualt_file=Main(start=50,end=55,interval=1)
# acc4=evaluate_mode(resualt_file)
# resualt_file=Main(start=60,end=65,interval=1)
# acc5=evaluate_mode(resualt_file)
# resualt_file=Main(start=65,end=70,interval=1)
# acc6=evaluate_mode(resualt_file)
# resualt_file=Main(start=75,end=80,interval=1)
# acc7=evaluate_mode(resualt_file)
# resualt_file=Main(start=80,end=85,interval=1)
# acc8=evaluate_mode(resualt_file)



#resualt_file=Main(start=0,end=3,interval=1)