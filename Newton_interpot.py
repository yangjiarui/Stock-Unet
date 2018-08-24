#-*-coding:utf-8-*-
'''

@author:HANDSOME_JERRY
@time:'18-6-28下午7:37'
'''
import numpy as np
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel('/home/jerry/workspace/my_projects/MASTER_project/sp500_office_reit.xlsx')
df=df.iloc[:,1]

class Node(object):

    def __init__(self,data,y_referance):
        self.data=data
        self.left=data[:-1]
        self.right=data[1:]
        self.y_referance=y_referance
    def f(self,x):
        c1=np.sin(x) #+ np.random.randn(1)
        c2=x+np.array(3.12313)
        c3=np.random.randn(11)
        c3=np.array([c3[5]])
        df=self.y_referance[int([x][0])]
        return np.array([df])
    def unit(self):
        if len(self.data)==2:
            return (self.f(self.right)-self.f(self.left))/(self.data[-1]-self.data[0])
        else:
            print('error')

def method(data,y_data):
    if len(data)==2:
        cur=Node(data=data,y_referance=y_data)
        return cur.unit()
    else:
        cur=Node(data=data,y_referance=y_data)
        data_left,data_right=cur.left,cur.right
        if len(data_left)==2:
            pass
            # print('the last 2nd layer nodes: %s'%len(cur.data))
        else:
            pass
            # print('predicting %s'%len(cur.data))
        return (method(data= data_right,y_data=y_data)-method(data=data_left,y_data=y_data))/(cur.data[-1]-cur.data[0])



def evaluate(oringin,x,y_data):
    a=pd.Series(oringin).sort_index(ascending=False).values
    factor=1
    full=[]
    for seq,i in enumerate(oringin[:-1]):
        factor=factor*(x-i)
        item=factor*method(data=a[-(seq+2):],y_data=y_data)
        full.append(item)
    last=[x]
    for seq,c in enumerate(oringin.tolist()):
        last.append(c)
    last = pd.Series(last).sort_index(ascending=False).values
    # final=Node(data=np.array([1,2]),y_referance=y_data).f(oringin[0])+sum(full)+factor*(x-oringin[-1])*method(data=last,y_data=y_data)
    final=Node(data=np.array([1,2]),y_referance=y_data).f(oringin[0])+sum(full)
    #print('oringin',Node(data=np.array([1,2])).f(oringin))
    # print('my',final)

    #a = Node(data=np.array([1,2]),y_referance=df.values).f(oringin).tolist()
    a=Node(data=np.array([1,2]),y_referance=df.values).y_referance.tolist()
    a.append(final)
    if a[-1]>min(a[:-1])-np.mean(a[:-1])+1.65*np.std(a[:-1]) and a[-1]<max(a[:-1])+np.mean(a[:-1])+1.65*np.std(a[:-1]):
        print('contains good pridect')
        return a[-1]
    elif a[-1] is not None:
        #print('not good')
        return a[-1]
    else:
        print('find another filter method')
# def normallize_newton(oringin,num):
#     possible1,possible2=[],[]
#     for i,j in zip(oringin[:num],oringin[num+1:]):   # 3 must < num
#         q1,q2=evaluate(oringin=oringin,x=i),evaluate(oringin=oringin,x=j)
#         if q1 != None:
#             possible1.append(q1)
#         if q2!=None:
#             possible2.append(q2)
#         else:
#             continue
#     if len(possible1+possible2)==0:
#         print('could not solve out ')
#     else:
#         pos=possible1+possible2
#         pos=set(pos)
#         pos=list(pos)
#         pos.sort()
#         return np.mean(pos[:int(0.75*len(pos))])


# datax=np.linspace(1,5,50)
# datay=np.sin(datax)
# nod=5+5./50
# q=evaluate(oringin=datax[:10],x=nod)
# print('q:%s'%q)
# plt.scatter(datax,datay,label='true')
# plt.scatter(nod,q,label='predict_values')
# plt.show()
#
def find_y_accroding_to_x(Num,y_data):
    Num=Num
    datax=np.array(range(Num-5,Num)).tolist()+np.array(range(Num+1,Num+5)).tolist()
    # datay=y_data[Num-5:Num].tolist()+y_data[Num+1:Num+5].tolist()
    q=evaluate(oringin=np.array(datax),x=Num,y_data=y_data)
    # plt.scatter(datax,datay)
    # plt.scatter(Num,q)
    return q


def isNum2(value):
    try:
        x = int(value)
    except TypeError:
        return False
    except ValueError:
        return False
    except Exception:
        return False
    else:
        return True
#
#
# nul=[]
# havedone=0
# ds=df.copy()
# for j in range(len(df)):
#     if isNum2(df[j]) == False:
#         nul.append(j)
#         if str(find_y_accroding_to_x(Num=j,y_data=df.values)[0])!='nan':
#             havedone+=1
#         lis=[df[j-1],df[j+1],find_y_accroding_to_x(Num=j,y_data=df.values)[0]]
#         df[j]=np.mean([q for q in lis if str(q)!='nan'])
#
# print('have predicted Done:',float(havedone/len(nul)))
# # df=df.fillna(method='ffill')
# # df=df.fillna(method='bfill')
# print(df.isnull().any())












