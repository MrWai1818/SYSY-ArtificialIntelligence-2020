import copy
import time
import heapq
'''

！！！该文件为最终版本，在As1.2的基础上进行如下修改：
    1.改进扩展节点的mov函数，使节点扩展从2层循环嵌套改进
    为单层循环
    2.close集的存储结构改成set()，以减少流程图中的一个
    判断步骤
    ！3.As1.2双向A*在open集寻找相同元素需要双重循环，
    对于更多步骤较大的案例，时间复杂度过高。使用dopp、opp
    两个set()简化判断条件，以达到不需要循环即可解答，
    时间复杂度从O(n^2)->O(1)。


'''
def printout(proad) :
    for i in proad :
        for j in range(16) :
            if j%4 ==3 :
                print(i[j])
            else :
                print(i[j],end=' ')     
        print('---')
def doAs(startp) :
    final = ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','0')
    road = []   #起点到交界输出路径
    droad = []  #终点到交接输出路径
    op = []     
    dop = []    
    opp = set() #新设置用于判断两个open相同节点的set
    dopp = set()
    cl = set()
    dcl = set() 
    class node(object):
        def __init__(self,pre,puzzle,f,g,h):
            self.puzzle = puzzle    #节点状态
            self.pre = pre          #前继节点
            self.f = f              #对应f值
            self.g = g              #g值
            self.h = h              #h值
        def getf(self,t) :
            num = 0 
            for i in range(16) :
                for j in range(16) :
                    if (self.puzzle[i] == t[j])&(self.puzzle[i] != '0') :
                        num +=  abs(i%4 - j%4) + abs(i//4 - j/4)    #计算曼哈顿距离
            self.h = 4*num
            self.g += 1
            self.f = self.h/3 + self.g  
            
        def __lt__(self, node): # 重载heapq的比较函数
            if self.f == node.f:
                return self.g> node.g
            return self.f < node.f  
                
    def move(p) :   #将所有扩展先存入列表，之后再在A*算法中判断是否重复
        mov =[]     #可减少一层循环
        i = p.index('0')  
        if  (i > 3) :   #仅判断是否到达边界，直接进行扩展
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
        cnt = 0
        while  (len(op) != 0) & (len(dop) != 0) :

            cnt += 1            #扩展结点数计算
            cur = heapq.heappop(op)     #取得open中最优节点并将其从open删除
            cur1 = heapq.heappop(dop)
            
            opp.remove(cur.puzzle)      #删除最优点，需要与两个open同步
            dopp.remove(cur1.puzzle)
            
            temp =move(cur.puzzle)      #存放扩展节点
            temp1 = move(cur1.puzzle)
            
            if opp&dopp !=set() :       #判断两个open是否有相同节点
                return opp&dopp,cnt
            
            for i in range(len(temp)) : #对扩展结点进行判断
                if (temp[i] in cl) :    #若重复则跳过
                    continue
                cl.add(temp[i])         #新的节点则加入open和close
                tn1 = node(cur, temp[i], cur.f, cur.g, cur.h)
                tn1.getf(final)
                heapq.heappush(op, tn1)
                opp.add(temp[i])        #同步
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
    same1,cnt = Astar(startp)
    same =same1.pop()
    time2 = time.time()
    printp1(findN(same, op), road)
    printp1(findN(same, dop), droad)
   
    newpath = makewhole(road, droad,same)
    return newpath,time1,time2,cnt
    
if __name__ == '__main__':
    startp=tuple(input().split())
    
    newpath,time1,time2,n=doAs(startp)
    
    printout(newpath)
    print(len(newpath)-1)
    print("timing: ",time2-time1)
    print("numbers of expanded node :",n)
