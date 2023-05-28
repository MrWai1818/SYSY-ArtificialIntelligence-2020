import copy
import time
import heapq
'''

该文件为对As1.1的修改：1.在双向A*基础上改变数据存储结构，
使用元组节点状态。
2.将open集改成优先队列heapq，取代原本列表的sort函数

'''
cnt = 0
final = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0')
class node(object):
    def __init__(self,pre,premov,puzzle,f,g,h):
        self.puzzle = puzzle    #
        self.pre = pre
        self.premov = premov
        self.f = f
        self.g = g
        self.h = h
    def getf(self,t) :
        num = 0 
        for i in range(16) :
            for j in range(16) :
                if (self.puzzle[i] == final[j])&(self.puzzle[i] != '0') :
                    num += abs(i%4 - j%4) + abs(i//4 - j//4)#将坐标距离整除改为正常除法
        self.h = 4*num         #h取值乘上4/3 
        self.g += 1
        self.f = self.h/3 + self.g  
    def __lt__(self, node): # 重载heapq的比较函数
        if self.f == node.f:
            return self.g> node.g
        return self.f < node.f             
def move(cur,how,t) :
    afmov = copy.deepcopy(cur) 
    afmov.pre = cur
    if (afmov.premov!= 'down')&(how == 'up') :
        afmov.premov = 'up'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i > 3) :
                afmov.puzzle = list(afmov.puzzle)
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i-4]
                afmov.puzzle[i-4] = temp
                afmov.puzzle = tuple(afmov.puzzle)
                afmov.getf(t)
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
                afmov.getf(t)
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
                afmov.getf(t)
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
                afmov.getf(t)
                return afmov
        return 0
    else :
        return 0
road = []
droad = []
op = []
dop = []
cl = []
dcl = []
def comp (t1,s,t) :
    for i in s :
        if (t1.puzzle == i.puzzle)&(t1.g < i.g) :
            s.remove(i)
            t.append(t1)
            return 1
    return 0
def Astar(sp) :
    while True :
        global cnt
        cnt+=1
        if len(op) == 0 :
            return 0
        if len(dop) == 0 :
            return 0
        cur = heapq.heappop(op)#选取f值最小节点同时从open表中删除
        cur1 = heapq.heappop(dop)
        howtp = ['up','down','left','right']
        for i in howtp :
            temp = move(cur, i, final)
            temp1 = move(cur1, i, sp)
            if (temp==0)&(temp1==0) :
                continue
            elif (temp1==0)&(temp!=0) :
                if (comp(temp, op,op)==0)&(comp(temp, cl,op) ==0) :
                    heapq.heappush(op, temp)
            elif (temp==0)&(temp1!=0) :
                if (comp(temp1, dop,dop)==0)&(comp(temp1, dcl,dop) ==0) :
                    heapq.heappush(dop, temp1)
            else :
                if (comp(temp, op,op)==0)&(comp(temp, cl,op) ==0) :
                    heapq.heappush(op, temp)
                if (comp(temp1, dop,dop)==0)&(comp(temp1, dcl,dop) ==0) :
                    heapq.heappush(dop, temp1)
        for i in op :
            for j in dop :
                if i.puzzle == j.puzzle :
                    cl.append(cur)
                    cl.append(i)
                    dcl.append(cur1)
                    return
        cl.append(cur)
        dcl.append(cur1)
            
def findrd() :
    i = cl[-1]
    while i in cl :
        road.append(i.puzzle)
        if(i.pre != '') :
            i = i.pre
        else :
            return
def dfindrd() :
    i = dcl[-1]
    while i in dcl :
        droad.append(i.puzzle)
        if(i.pre != '') :
            i = i.pre
        else :
            return
def printout() :
    for i in road[::-1] :
        for j in range(16) :
            if j%4 ==3 :
                print(i[j])
            else :
                print(i[j],end=' ')     
        print('---')
def dprintout() :
    for i in droad :
        for j in range(16) :
            if j%4 ==3 :
                print(i[j])
            else :
                print(i[j],end=' ')     
        print('---')
startp=tuple(input().split())
start = node('', '',startp, 0, -1, 0)
start.getf(final)
op.append(start)
finalo = node('', '', final, 0, -1, 0)
finalo.getf(startp)
dop.append(finalo)
time1 = time.time()
Astar(startp)
time2 = time.time()
findrd()
dfindrd()
printout()
dprintout()
print(len(road)+len(droad)-1)
print(cnt)
print("timing: ",time2-time1)
