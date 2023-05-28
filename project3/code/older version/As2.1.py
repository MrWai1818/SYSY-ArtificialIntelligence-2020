import copy
import time
import heapq
'''

该文件在As2_0的基础上，进一步对节点状态进行压缩，
将所有节点信息放在一个tuple中

'''
final = (0,-1,'','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0')#f|g|pre
road = []
droad = []
op = []
dop = []
opp = set()
dopp = set()
cl = set()
dcl = set()
nfn = 0
def getf(s,t) :
    num = 0 
    g = 0 
    for i in range(3,19) :
        for j in range(3,19) :
            if (s[i] == t[j])&(s[i] != '0') :
                num +=  abs(i%4 - j%4) + abs(i//4 - j/4)
    h = 4*num
    g += 1
    f = h/3 + g  
    s = list(s)
    s[0] = f
    s[1] = g
    return tuple(s) 
              
def move(p) :
    mov =[]
    i = p.index('0')
    if  (i > 6) :
        t = list(p)        
        t[i] = t[i-4]
        t[i-4] = '0'
        t[2] = p
        t = tuple(t)
        mov.append(t)
    if  (i < 15 ) :
        t = list(p)        
        t[i] = t[i+4]
        t[i+4] = '0'
        t[2] = p
        t = tuple(t)
        mov.append(t)
    if  ((i-3)%4 > 0) :
        t = list(p)        
        t[i] = t[i-1]
        t[i-1] = '0'
        t[2] = p
        t = tuple(t)
        mov.append(t)
    if  ((i-3)%4 < 3) :
        t = list(p)        
        t[i] = t[i+1]
        t[i+1] = '0'
        t[2] = p
        t = tuple(t)
        mov.append(t)
    return mov
def Astar(sp) :
    while  (len(op) != 0) & (len(dop) != 0) :
        global nfn 
        nfn += 1
        print(nfn)
        cur = heapq.heappop(op)
        cur1 = heapq.heappop(dop)
        opp.remove(cur[3:])
        dopp.remove(cur1[3:])
        temp =move(cur)
        temp1 = move(cur1)
        
        if opp&dopp !=set() :
            return opp&dopp
        
        for i in temp :
            if (i[3:] in cl) :
                continue
            cl.add(i[3:])
            tn1 = getf(i,final)
            heapq.heappush(op, tn1)
            opp.add(i[3:])
        for i in temp1 :
            if(i[3:] in dcl) :
                continue
            dcl.add(i[3:])
            tn2 = getf(i,sp)
            heapq.heappush(dop, tn2)
            dopp.add(i[3:])
def findN(t,where) :
    for i in where :
        if i[3:]==t :
            return i
def printp1(p,proad) :
    if p[2] !='' :
        printp1(p[2],proad)
    proad.append(p[3:])
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
    start = (0,-1,'')+startp
    
    finalo = getf(final, start)
    starto = getf(start, final)
    
    cl.add(startp)
    dcl.add(final)

    op.append(starto)
    dop.append(finalo)

    opp.add(start[3:])
    dopp.add(final[3:])
    
    heapq.heapify(op)
    heapq.heapify(dop)

    time1 = time.time()
    same = Astar(start).pop()
    time2 = time.time()

    printp1(findN(same, op), road)
    printp1(findN(same, dop), droad)
   
    newpath = makewhole(road, droad,same)
    printout(newpath)
    print(len(newpath)-1)
    print("timing: ",time2-time1)
    print("numbers of expanded node :",nfn)
