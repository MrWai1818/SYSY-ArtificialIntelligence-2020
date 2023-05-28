import matplotlib.pyplot as plt 
import random,math
import time
totald = []
totalt = []
def mk_cord() : #输入数据
    node_num = int(input())
    cord = []
    for i in range(node_num) :
        get_node = input().split()
        cord.append(get_node[1:])  
    return node_num,cord
def get_adjm(node_num,cord) :    #创建邻接矩阵
    adm = list(0.0 for index in range(node_num**2))
    for i in range(node_num) :
        for j in range(i,node_num) :    #计算欧氏距离
            adm[i*node_num + j] = adm[j*node_num + i] = ((float(cord[i][0])-float(cord[j][0]))**2 + (float(cord[i][1])-float(cord[j][1]))**2)**0.5
    return adm    
def print_matrix(node_num,adm) :    #打印邻接矩阵
    for i in range(node_num **2) :
        if (i+1)%node_num == 0 :
            print(adm[i])
        else :
            print(adm[i],end='  ')
def draw_road(cord,bestroad) :  #画出城市路线图
    xp = []
    yp = []
    for i in bestroad :     #存好每个点的数据
        xp.append(float(cord[i][0]))
        yp.append(float(cord[i][1]))
    plt.plot(xp,yp,'g-s')   #标每个点的数据
    x = [xp[0],xp[-1]]
    y = [yp[0],yp[-1]]
    plt.plot(x,y,"*b:",markersize=10)   #起点终点特别标注
    plt.show()
  
def reverse_segment(p,node_num) :#倒转两个点之间的所有点顺序
    new = random.sample(p, 2)   #随机取两个点
    new.sort()              #排序
    temp = p[0:new[0]]      #截取到选中点前    
    temp_re = list(reversed(p[new[0]:new[1]]))  #逆转顺序
    temp.extend(temp_re)    
    temp.extend(p[new[1]:node_num])
    return temp
def exchange_node(dad,mom,node_num) :#将两条路径某一点后的所有节点交换
    pos = random.randint(0, node_num)#随机取点
    son1 = dad[0:pos]           #两天路径slice
    son2 = mom[0:pos]
    son1.extend(mom[pos:node_num])#交换随机点后的所有节点
    son2.extend(dad[pos:node_num])
    same1 = []  #查重
    same2 = []
    for i in range(pos,node_num):   #找到重复进行交换
        for j in range(pos):
            if son1[i] == son1[j]:
                same1.append(j)
            if son2[i] == son2[j]:
                same2.append(j)
    for i in range(len(same1)):
        son1[same1[i]], son2[same2[i]] = son2[same2[i]], son1[same1[i]]
    return son1,son2 

def exchange_double_node(dad,mom,node_num) :#将两条路径某一点后的所有节点交换
    pos = random.sample(dad, 2)#随机取点
    pos.sort()
    son1 = dad[0:pos[0]]           #两天路径slice
    son2 = mom[0:pos[0]]
    t1 = dad[pos[0]:pos[1]]
    t2 = mom[pos[0]:pos[1]]
    son1.extend(mom[pos[0]:pos[1]])#交换随机点后的所有节点
    son2.extend(dad[pos[0]:pos[1]])
    son1.extend(dad[pos[1]:node_num])
    son2.extend(mom[pos[1]:node_num])
    same1 = []  #查重
    same2 = []
    for i in range(pos[0],pos[1]):   #找到重复进行交换
        for j in range(0,pos[0]):
            if son1[i] == son1[j]:
                same1.append(j)
            if son2[i] == son2[j]:
                same2.append(j)
    for i in range(pos[0],pos[1]):  
        for j in range(pos[1],node_num):
            if son1[i] == son1[j]:
                same1.append(j)
            if son2[i] == son2[j]:
                same2.append(j)
    for i in range(len(same1)):
        son1[same1[i]], son2[same2[i]] = son2[same2[i]], son1[same1[i]]
    
    return son1,son2 

def insert_segment(p,node_num) :#将两个节点中间所有节点插入第三个节点后
    new = random.sample(p, 3)   #随机取三个点
    new.sort(reverse=True)      #排序
    temp = p[new[2]:new[1]].copy()      #交换
    p[new[2]:new[0]+new[2]-new[1]+1] = p[new[1]:new[0]+1].copy()
    p[new[0]+new[2]-new[1]+1:new[0]+1] = temp   #插入
    return p
def waySA(node_num,cord) :
    adm = get_adjm(node_num,cord)
    #print_matrix(node_num,adm)   #按照矩阵打印列表
    res = []
    newroad = list(range(node_num)) #产生新解的路径，也是初始化
    random.shuffle(newroad)         #随机打乱
    curroad = newroad.copy()        #局部最优解
    bestroad = newroad.copy()       #全局最优解
    bestdist = 9000000          #全局最小距离
    curdist = 9000000           #局部最小距离
    
    speed = 0.99                #退火速度
    t_min = 0.0000001           #结束温度
    t_cur = 10000               #初温
    while t_cur > t_min :       #循环
        for i in range(3000) :  #达到稳定前进行循环(产生局部最优解前)
            random_num = random.random()    #3等分概率，随机数决定使用哪种方式改变路径
            if 0.6 > random_num > 0.3 :     #直接交换两个点的位置
                new = random.sample(newroad, 2)
                newroad[new[0]],newroad[new[1]] = newroad[new[1]],newroad[new[0]]
            elif random_num >= 0.6 :        #将两个点间的所有节点加入第三个点后
                newroad = insert_segment(newroad, node_num)
            else :                  #逆转两个点间的所有节点
                newroad = reverse_segment(newroad, node_num)
            newdist = 0             #新解的路径长度
            for i in range(node_num) :  #计算新解路径长度
                if i!=node_num-1 :
                    newdist += adm[newroad[i]*node_num+newroad[i+1]]
                else :
                    newdist += adm[newroad[0]*node_num+newroad[-1]]
            if newdist < curdist :      #如果新解优于当前最优解则替换
                curdist = newdist
                curroad = newroad.copy()
                if newdist <bestdist :  #如果新解优于全局最优解则替换
                    bestdist = newdist
                    bestroad = newroad.copy()
            else :  #否则按照一定概率接受新解
                if random.random() < math.exp(-(newdist-curdist)/t_cur) :
                    curdist = newdist
                    curroad = newroad.copy()
                else :
                    newroad = curroad.copy()
        t_cur = t_cur*speed #退火温度降低
        res.append(bestdist)
        print(t_cur)
    print('The shortest distance : ',bestdist)
    print('The shortest road : ',bestroad)
    draw_road(cord, bestroad)   #画出退火曲线
    plt.plot(res)
    totald.append(bestdist)
    plt.xlabel('annealing times')
    plt.ylabel('best_result')
    plt.show()

def wayGA(node_num,cord) :
    adm = get_adjm(node_num,cord)
    races = []      #种群
    species = 5000 #物种
    num = 500   #群落
    individual = 0 #个体
    res = []
    ancestor = list(range(node_num)) #祖先
    cnt = 0
    while cnt != num :   #初始化种群
        random.shuffle(ancestor)    #随机改变
        temp = ancestor.copy()
        if temp not in races :
            races.append(temp)
            cnt = cnt +1
    ind_list = []
    ind_dis = []
    while individual < species :    #开始进行遗传
        individual += 1     #遗传代数++
        f = []
        #获取适应度函数
        for j in races :    
            newdist = 0
            for i in range(node_num) :#计算种群中每条路径的长度
                if i!=node_num-1 :
                    newdist += adm[j[i]*node_num+j[i+1]]
                else :
                    newdist += adm[j[0]*node_num+j[-1]]
            f.append(1/newdist)#压入
        sig = sum(f)
        #选择
        p = []
        new_races = []  #新种群
        #锦标赛选择
        # for _ in range(num):  
        #     o = list(range(num))
        #     i, j = random.sample(o, 2)    #随机取两条路径进行比较
        #     temp_i_dist ,temp_j_dist= 0,0     
        #     for k in range(node_num) :    #计算第一条路径长度
        #         if k!=node_num-1 :
        #             temp_i_dist += adm[races[i][k]*node_num+races[i][k+1]]
        #         else :
        #             temp_i_dist += adm[races[i][0]*node_num+races[i][-1]]
        #     for k in range(node_num) :    #计算第二天路径长度
        #         if k!=node_num-1 :
        #             temp_j_dist += adm[races[j][k]*node_num+races[j][k+1]]
        #         else :
        #             temp_j_dist += adm[races[j][0]*node_num+races[j][-1]]
        #     if temp_i_dist < temp_j_dist: #路径长度短的压入新种群
        #         new_races.append(races[i])
        #     else:
        #         new_races.append(races[j])

        #轮转赌选择
        for i in f :        #利用适应度函数计算出判断标准p
            p.append(i/sig)
        temp = {}       
        for i in range(len(p)): #先取出种群中最好的前百分之20条路径
            temp[i] = p[i]
        index_of_max = [k[0] for k in sorted(temp.items(),key=lambda e:e[1])[::-1][:int(num*0.2)]]
        
        for i in index_of_max :     #压入新种群
            new_races.append(races[i])
        temp_index = list(range(num))
        choose = random.choices(temp_index,weights=p,k=int(0.8*num))    #随机从剩下的路径出成员组成新种群
        for i in choose :
            new_races.append(races[i])
        #开始遗传
        for i in range(int(num/2)) :
            j = i + int(num/2) - 1
            if random.random() < 0.5 :#交叉
                mom = new_races[i]
                dad = new_races[j]
                son1,son2 = exchange_node(dad, mom, node_num)
                if random.random() <0.2:#变异
                    temp_rand = random.random()
                    # if  temp_rand <0.8 :#变异形式1，翻转两个结点间的内容
                    son1 = reverse_segment(son1, node_num)
                    son2 = reverse_segment(son2, node_num)
                    # elif  temp_rand < 0.2:#变异形式2，将两个结点间内容插入第三个结点后
                    #     son1 = insert_segment(son1, node_num)
                    #     son2 = insert_segment(son2, node_num)
                    # else :#变异形式3，随机打乱结点
                    # random.shuffle(son1)
                    # random.shuffle(son2)
                new_races[i] = son1
                new_races[j] = son2
        races = new_races
        dist = []   #种群距离
        for j in races :   
            newdist = 0
            for i in range(node_num) :
                if i!=node_num-1 :
                    newdist += adm[j[i]*node_num+j[i+1]]
                else :
                    newdist += adm[j[0]*node_num+j[-1]]
            dist.append(newdist)
        print(individual)
        bestdist = min(dist)
        res.append(bestdist)
        best_index = dist.index(bestdist)
        bestroad = races[best_index]  
    print('The shortest distance : ',bestdist)
    totald.append(bestdist)
    print('The shortest road : ',bestroad)
    draw_road(cord, bestroad)
    plt.xlabel('generation')
    plt.ylabel('best_result')
    plt.plot(res)
    plt.show()    

def main() :
    while 1 :
        print(80*"=")
        print("Input '1' to call SA     Input '2' to call GA     Input '0' to exit")
        which = input()
        if which == '1' :
            print("Please input the citys' data")
            node_num ,cord= mk_cord()    #结点数以及结点矩阵
            time1 =time.time()
            waySA(node_num,cord)
            time2 = time.time()
            print("cost time :",time2-time1)
        elif which == '2' :
            print("Please input the citys' data")
            node_num ,cord= mk_cord()    #结点数以及结点矩阵
            time1 =time.time()
            wayGA(node_num,cord)
            time2 = time.time()
            print("cost time :",time2-time1)
        elif which == '0' :
            print("Thank u")
            break
        else :
            print("Wrong input,please input again")    
if __name__ == '__main__':
    # node_num ,cord= mk_cord()    #结点数以及结点矩阵
    # for i in range(5) :
    #     time1 =time.time()
    #     #waySA(node_num,cord)
    #     wayGA(node_num,cord)
    #     time2 = time.time()
    #     print("cost time :",time2-time1)
    #     totalt.append(time2-time1)
    # print(totald)
    # print(totalt)
    # print("1")
    main()