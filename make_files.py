#-*-coding:utf-8-*-
import pandas as pd
import os
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')



j=0
names=os.listdir('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/a_all_data')[:100]
for i in names:
    new=pd.DataFrame()
    df=pd.read_csv('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/a_all_data/'+i)
    if len(df) <4096:
        j+=1
        print('drop_out:',i)
    else:
        new['time']=df.iloc[:,2]
        new['name']=df.iloc[:,0]
        nod=[]
        #nod.append(255)
        close=df.iloc[:,3]
        #for j in range(len(close)-1):
        #    if close[j+1]>=close[j]:
        #        nod.append(255)
        #    else:
        #        nod.append(0)
        #nod.append(255)
        new['close']=close
        q=i.split('.')
        cur=q[0]+q[1]
        new=new.sort_index(ascending=False)
        new.to_csv('/home/jerry/PycharmProjects/untitled/venv/MASTER_project/new_a_data/a+'+q[0]+q[1]+'.csv')
print('Done')
print('has_droped:',len(names)-j)
