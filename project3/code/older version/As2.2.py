import copy
import time
import heapq
'''

该文件在As2_0的基础上，对启发式函数的运算方式进行修改

'''
final = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0')
road = []
droad = []
op = []
dop = []
opp = set()
dopp = set()
cl = set()
dcl = set()
nfn = 0
class node(object):
    def __init__(self,pre,puzzle,f,g,h):
        self.puzzle = puzzle    
        self.pre = pre
        self.f = f
        self.g = g
        self.h = h
    def getf(self,t) :
        num = 0 
        for i in range(16):#使用单层循环进行运算
            if (self.puzzle[i] == t[i]) |( self.puzzle[i]== 0):
                continue
            else:
                x =  (int(self.puzzle[i]) - 1) // 4  
                y =  int(self.puzzle[i]) - 4 * x - 1
                num += (abs(x - i /4) + abs(y - i % 4))
        self.h = 4*num
        self.g += 1
        self.f = self.h/3 + self.g  
        
    def __lt__(self, node): # 重载heapq的比较函数
        if self.f == node.f:
            return self.g> node.g
        return self.f < node.f  
               
def move(p) :
    mov =[]
    i = p.index('0')
    if  (i > 3) :
        t = list(p)        
        t[i] = t[i-4]
        t[i-4] = '0'
        t = tuple(t)
        mov.append(t)
    if  (i < 12 ) :
        t = list(p)        
        t[i] = t[i+4]
        t[i+4] = '0'
        t = tuple(t)
        mov.append(t)
    if  (i%4 > 0) :
        t = list(p)        
        t[i] = t[i-1]
        t[i-1] = '0'
        t = tuple(t)
        mov.append(t)
    if  (i%4 < 3) :
        t = list(p)        
        t[i] = t[i+1]
        t[i+1] = '0'
        t = tuple(t)
        mov.append(t)
    return mov
def Astar(sp) :
    while  (len(op) != 0) & (len(dop) != 0) :
        global nfn 
        nfn += 1
        cur = heapq.heappop(op)
        cur1 = heapq.heappop(dop)
        
        opp.remove(cur.puzzle)
        dopp.remove(cur1.puzzle)
        
        temp =move(cur.puzzle)
        temp1 = move(cur1.puzzle)
        
        if opp&dopp !=set() :
            return opp&dopp
          
        for i in range(len(temp)) :
            if (temp[i] in cl) :
                continue
            cl.add(temp[i])
            tn1 = node(cur, temp[i], cur.f, cur.g, cur.h)
            tn1.getf(final)
            heapq.heappush(op, tn1)
            opp.add(temp[i])
        for i in range(len(temp1)) :
            if(temp1[i] in dcl) :
                continue
            dcl.add(temp1[i])
            tn2 = node(cur1, temp1[i], cur1.f, cur1.g, cur1.h)
            tn2.getf(sp)
            heapq.heappush(dop, tn2)
            dopp.add(temp1[i])
def findN(t,where) :
    for i in where :
        if i.puzzle==t :
            return i
def printp1(p,proad) :
    if p.pre !='' :
        printp1(p.pre,proad)
    proad.append(p.puzzle)
def makewhole(s,p,same) :
    if (len(s)+len(p))%2!=0 :
        s.remove(same)
        p.remove(same)
    for i in s :
        if i in p :
            p.remove(i)
    return s+p[::-1]
def printout(proad) :
    for i in proad :
        for j in range(16) :
            if j%4 ==3 :
                print(i[j])
            else :
                print(i[j],end=' ')     
        print('---')
if __name__ == '__main__':
    startp=tuple(input().split())
    start = node('',startp, 0, -1, 0)
    start.getf(final)
    finalo = node('', final, 0, -1, 0)
    finalo.getf(startp)
    cl.add(startp)
    dcl.add(final)

    op.append(start)
    dop.append(finalo)
    
    opp.add(startp)
    dopp.add(final)
    
    heapq.heapify(op)
    heapq.heapify(dop)

    time1 = time.time()
    same = Astar(startp).pop()
    time2 = time.time()

    printp1(findN(same, op), road)
    printp1(findN(same, dop), droad)
   
    newpath = makewhole(road, droad,same)
    printout(newpath)
    print(len(newpath)-1)
    print("timing: ",time2-time1)
    print("numbers of expanded node :",nfn)
