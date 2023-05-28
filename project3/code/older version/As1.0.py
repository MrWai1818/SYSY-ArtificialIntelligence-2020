import copy
import time
'''

该文件为最初的A*算法实现

'''
cnt = 0
final = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0')
class node(object):
    def __init__(self,pre,premov,puzzle,f,g,h):
        self.puzzle = puzzle    #节点状态
        self.pre = pre          #前继节点
        self.premov = premov    #前继节点的移动
        self.f = f              #以下对应f、g、h
        self.g = g
        self.h = h
    def getf(self) :            #计算曼哈顿距离
        num = 0 
        for i in range(16) :
            for j in range(16) :
                if (self.puzzle[i] == final[j])&(self.puzzle[i] != '0') :
                    num += abs(i%4 - j%4) + abs(i//4 - j//4)
        self.h = num
        self.g += 1
        self.f = self.h + self.g                  
def move(cur,how) :     #对节点进行扩展
    afmov = copy.deepcopy(cur) 
    afmov.pre = cur         #设置前继节点
    if (afmov.premov!= 'down')&(how == 'up') : #判断前继节点的移动，以免重复
        afmov.premov = 'up'
        for i in range(16) :        #界面大小为4*4，所以0点移动要判断边界
            if (afmov.puzzle[i] == '0')&(i > 3) :  #找到‘0’点，判断是否在上边界
                afmov.puzzle = list(afmov.puzzle)   #即第一行
                temp = afmov.puzzle[i]     #该函数之后的elif均以该部分为模板
                afmov.puzzle[i] = afmov.puzzle[i-4]#仅是改动数据以及判断边界
                afmov.puzzle[i-4] = temp    #便不再赘述
                afmov.puzzle = tuple(afmov.puzzle)
                afmov.getf()
                return afmov
        return 0
    elif (afmov.premov!= 'up')&(how == 'down') :    
        afmov.premov = 'down'       
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i < 12) :
                afmov.puzzle = list(afmov.puzzle)
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i+4]
                afmov.puzzle[i+4] = temp
                afmov.puzzle = tuple(afmov.puzzle)
                afmov.getf()
                return afmov
        return 0
    elif (afmov.premov!= 'right')&(how == 'left') :
        afmov.premov = 'left'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i%4 > 0) :
                afmov.puzzle = list(afmov.puzzle)
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i-1]
                afmov.puzzle[i-1] = temp
                afmov.puzzle = tuple(afmov.puzzle)
                afmov.getf()
                return afmov
        return 0
    elif (afmov.premov!= 'left')&(how == 'right') :
        afmov.premov = 'right'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i%4 < 3) :
                afmov.puzzle = list(afmov.puzzle)
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i+1]
                afmov.puzzle[i+1] = temp
                afmov.puzzle = tuple(afmov.puzzle)
                afmov.getf()
                return afmov
        return 0
    else :
        return 0
road = []
op = []
cl = []
def comp (t1,s) :#检查扩展节点是否已经存在于open/close中
    for i in s :
        if (t1.puzzle == i.puzzle)&(t1.f < i.f) :#若存在且f值更小
            s.remove(i)     #则删除旧节点
            s.append(t1)    #更新f更小的节点
            return 1
    return 0
def Astar() :
    while True :
        global cnt
        cnt+=1
        if len(op) == 0 : #判断open集是否为空
            return 0
        cur = op[0]         #取open集f值最小的节点当前节点
        if cur.puzzle == final :    #算法成功
            cl.append(cur)
            return
        howtp = ['up','down','left','right']    #扩展结点的4种方式
        for i in howtp :        #进行节点扩展
            temp = move(cur, i) #根据mov函数得到全新的节点
            if temp==0 :
                continue
            if (comp(temp, op)==0)&(comp(temp, cl) ==0) :#对得到的新节点进行查重
                op.append(temp)
        op.remove(cur)  #删除open中当前节点
        cl.append(cur)  #将当前节点加入close
        op.sort(key=lambda node: node.f) #对open表按照f值排序、使节点按照升序存储
        #保证open表第一个节点为f值最小的节点
def findrd() :  #A*搜索完成后，起始节点到目标节点的完整路径就存在于close中
    i = cl[-1]  #对目标节点的前继节点进行倒序查找，即为路径
    while i in cl :
        if(i.pre != '') :
            road.append(i.puzzle)
            i = i.pre
        else :
            return
def printout() :
    cnt = 0
    for i in road[::-1] :
        for j in range(16) :
            if j%4 ==3 :
                print(i[j])
            else :
                print(i[j],end=' ')     
        print('---')
        cnt+=1
    print(cnt)
startp=input().split()
start = node('', '',startp, 0, 0, 0)
start.getf()
op.append(start)
time1 = time.time()
Astar()
time2 = time.time()
findrd()
printout()
print(cnt)
print("timing: ",time2-time1)