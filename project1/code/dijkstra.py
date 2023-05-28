def Dijkstra(pn,pr,ps,pe) :
    #p = input("请输入节点数和边数\n")
    m = pn.split()[0]
    n = pn.split()[1]
    #print("以下输入节点关系\n格式为'a b 12'，意为a到b的边权为12")
    gragh = {}    #图
    dist = {}    #到各个节点距离
    visit = []    #判断是否已经访问
    pre = {}   #每个节点最近的前驱节点
    road = []    #最短路径
    #制表
    def mkgraph(n,m) :
        temp = []
        for i in range(len(pr)) :   
            temp.append(pr[i].split())
            gragh[temp[i][0]] = {}
            gragh[temp[i][1]] = {}
            dist[temp[i][0]] = 9999
            dist[temp[i][1]] = 9999
        for relation in temp :
            gragh[relation[0]][relation[1]] = int(relation[2])
            gragh[relation[1]][relation[0]] = int(relation[2])
    mkgraph(n, m)
    #获取起始节点和目标节点
    #total = input("请输入起始节点和目标节点\n")
    #begin = total.split()[0]
    #end = total.split()[1]
    tdist = []
    d = {}
    for i in dist :
        tdist.append(i[0].title())
        d[i[0].title()] = i
    if (ps[0].title() not in tdist)|(pe[0].title() not in tdist) :    #防止输入不在图中的点
        dist.clear()
        road.append("Please input the correct vertex")
        return dist,road,' ',' '
    else :
        for k in d :
            if ps[0].title() == k :
                ps = d[k]
        for k in d :
            if pe[0].title() == k :
                pe = d[k]
    begin = ps
    end = pe
    dist.update(gragh[begin])
    visit.append(begin)
    for i in dist : 
        pre[i] = begin
    pre[begin] = None
    #dijkstra算法
    def dijkstra(dist,gragh) :
        for i in range(int(m)) :
            min_dist = 9999
            min_temp = begin
            for node in gragh : 
                if (node not in visit)&(dist[node] < min_dist) :
                    min_dist = dist[node]
                    min_temp = node
            visit.append(min_temp)
            for node in gragh[min_temp] :
                if  (gragh[min_temp][node]+dist[min_temp] <= dist[node])&(node != begin) :
                    dist[node] = gragh[min_temp][node]+dist[min_temp] 
                    pre[node] = min_temp 
        return dist
    #输出最短路径
    def putroad() :
        road = [end]
        put = pre[end]
        while put :
            road.append(put)
            put = pre[put]
        road.reverse()
        return road
    dist=dijkstra(dist, gragh) 
    #print("从 " + begin + " 到各个节点的距离为 "+ str(dist))
    #print("从 " + begin + " 到 " + end + " 的最短距离为 : " + str(dist[end]))
    #print("最短路径为 :",end=" ")
    road=putroad()
    return dist,road, begin, end