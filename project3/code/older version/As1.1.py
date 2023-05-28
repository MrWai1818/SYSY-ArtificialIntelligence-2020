import copy
import time
'''

该文件为对As1.0的修改：A*算法转化成双向A*

'''
cnt = 0
final = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0']   
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
                if (self.puzzle[i] == t[j])&(self.puzzle[i] != '0') :
                    num += abs(i%4 - j%4) + abs(i//4 - j//4)
        self.h = num*4
        self.g += 1
        self.f = self.h//3 + self.g                  
def move(cur,how,t) :
    afmov = copy.deepcopy(cur) 
    afmov.pre = cur
    if (afmov.premov!= 'down')&(how == 'up') :
        afmov.premov = 'up'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i > 3) :
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i-4]
                afmov.puzzle[i-4] = temp
                afmov.getf(t)
                return afmov
        return 0
    elif (afmov.premov!= 'up')&(how == 'down') :
        afmov.premov = 'down'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i < 12) :
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i+4]
                afmov.puzzle[i+4] = temp
                afmov.getf(t)
                return afmov
        return 0
    elif (afmov.premov!= 'right')&(how == 'left') :
        afmov.premov = 'left'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i%4 > 0) :
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i-1]
                afmov.puzzle[i-1] = temp
                afmov.getf(t)
                return afmov
        return 0
    elif (afmov.premov!= 'left')&(how == 'right') :
        afmov.premov = 'right'
        for i in range(16) :
            if (afmov.puzzle[i] == '0')&(i%4 < 3) :
                temp = afmov.puzzle[i]
                afmov.puzzle[i] = afmov.puzzle[i+1]
                afmov.puzzle[i+1] = temp
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
        cnt +=1
        if len(op) == 0 :
            return 0
        if len(dop) == 0 :
            return 0
        cur = op[0]         
        cur1 = dop[0]       #额外设置dop，第二个open表
        #两个open表的扩展同时进行
        howtp = ['up','down','left','right']
        for i in howtp :    #扩展函数同As1.0中实现，仅是同时扩展两个open
            temp = move(cur, i, final)
            temp1 = move(cur1, i, sp)
            if (temp==0)&(temp1==0) :
                continue
            elif (temp1==0)&(temp!=0) :
                if (comp(temp, op,op)==0)&(comp(temp, cl,op) ==0) :
                    op.append(temp)
            elif (temp==0)&(temp1!=0) :
                if (comp(temp1, dop,dop)==0)&(comp(temp1, dcl,dop) ==0) :
                    dop.append(temp1)
            else :
                if (comp(temp, op,op)==0)&(comp(temp, cl,op) ==0) :
                    op.append(temp)
                if (comp(temp1, dop,dop)==0)&(comp(temp1, dcl,dop) ==0) :
                    dop.append(temp1)
        op.sort(key=lambda node: node.f)    #同时重新排序两个open
        dop.sort(key=lambda node: node.f)
        for i in op :           #查找两个open表中的相同元素
            for j in dop :
                if i.puzzle == j.puzzle :
                    cl.append(cur)
                    cl.append(i)
                    dcl.append(cur1)
                    return
        op.remove(cur) #同时将不同当前节点压从对应open删除，加入对应close
        dop.remove(cur1)            
        cl.append(cur)   
        dcl.append(cur1)
        print(len(op))
            
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
startp=input().split()
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
