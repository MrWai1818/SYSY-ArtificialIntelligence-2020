import copy
def resolution(putin):
    class asgn :    #存放单个元素信息
        def __init__(self,name,var,tf) :
            self.var = []   #元素包含的变量
            self.tf = tf    #真值
            self.name = name    #元素的谓词名称

    class road :    #格式输出信息
        def __init__(self,bef,aft,fa,ma,w,where) :
            self.bef = bef  #变量替换前的参数
            self.aft = aft  #变量替换后的值
            self.fa = fa    #参与归结的子句之一
            self.ma = ma    #参与归结的子句之一
            self.w = []     #归结后子句
            self.where=where#按格式输出的行号
    var = []    #可赋给变量的值
    sen = []    #子句的集合
    rd = []     #归结出的结果
    jug = []    #判断子句是否能够归结
    frd = []    #最终路线
    outp=[]     #存放输出信息，方便ui设计
    osen=[]     #存放输出信息，方便ui设计
    def mkdict(putin) :
        tempp=putin.split('\n')
        p = int(tempp[0])
        tempp=tempp[1:]
        #以上为了ui输入方便
        for i in range(int(p)) :#循环输入子句
            tp = tempp[i].replace(' ', '')#之后的replace都是为了格式化输入，方便操作
            osen.append(tp)
            for j in range(tp.count(',')) :
                if tp[tp.find(',')-1] == ')' :#格式化输入
                    temp = list(tp)
                    temp[tp.find(',')] = '.'
                    tp = ''.join(temp)
                else :
                    temp = list(tp)
                    temp[tp.find(',')] = '/'
                    tp = ''.join(temp)
            if (tp[0]=='(') :#判断子句是否有多个元素
                tp = tp[1:len(tp)-1]
                stp = tp.split('.')
                num = []
                for k in stp :#格式化后生成对应元素的asgn列表
                    a1 = asgn('', '', True) 
                    if(k[0]=='¬') :
                        a1.tf = False
                        k = k[1:]
                    a1.name = k[0:k.find('(')]
                    pp = k[k.find("(")+1:k.find(')')]
                    a1.var =pp.split('/')
                    num.append(a1)#生成asgn列表，代表一个子句
                sen.append(num)#将子句压入
                jug.append(0)#判断子句的变量是未知量还是已知量
            else :#生成单个元素子句
                a1 = asgn('', '', True) 
                if(tp[0]=='¬') :
                    a1.tf = False
                    tp = tp[1:]
                a1.name = tp[0:tp.find('(')]
                pp = tp[tp.find("(")+1:tp.find(')')]
                a1.var =pp.split('/')
                for l in pp.split('/') :
                    var.append(l)
                u = []
                u.append(a1)
                sen.append(u)
                jug.append(1)#判断子句的变量是未知量还是已知量
    
    def do() : 
        while sen[len(sen)-1] :
            for i1 in sen :
                if jug[sen.index(i1)]==0 :  #定义jug函数判断i1的元素是已知量还是变量，若为变量则不能进行归结
                    continue
                for j1 in range(len(i1)) :  #因为子句中可能有多个元素，所以每个子句的元素需要二重循环来遍历
                    for i2 in sen[::-1] :#反向将每个子句一一对比，寻找归结对象
                        for j2 in range(len(i2)) :#下一行代码为找到可以进行最一般合一的单个元素
                                if (i1[j1].name == i2[j2].name)&(((i1[j1].tf==False)&(i2[j2].tf==True))|((i1[j1].tf ==True)&(i2[j2].tf ==False))) :
                                    if (len(i2[j2].var)>1) :    #进行元素中变量个数判断，元素变量个数大于一
                                        clock = 0               #以下判断该元素的变量是已知量还是未知数
                                        for jk in i2[j2].var :  #需要该元素变量含有未知数
                                            if jk not in var :
                                                clock =1        #或者全为已知量并且与i1中对应元素的变量值相同才可以合一
                                        if (clock ==1)|((clock==0)&(i1[j1].var==i2[j2].var)):         
                                            for h1 in range(len(i1[j1].var)) :      #在将归结的两个元素的变量中找到将赋值的未知量
                                                for h2 in range(len(i2[j2].var)) :
                                                    if i1[j1].var[h1]==i2[j2].var[h2] : #找到可以合一的元素变量
                                                        tnum = copy.deepcopy(i2)        #深拷贝来生成归结后子句，以免改变旧子句
                                                        tvar = copy.deepcopy(i1[j1].var) 
                                                        tnum[j2].var.remove(i1[j1].var[h1])
                                                        tvar.pop(h1)             #例子：i1[ji]=L(tony, rain)，i2=(¬C(y), ¬L(y, rain))       
                                                        ttt = tnum[j2].var[0]    
                                                        tnum.pop(j2)            #以上“确定”将tony赋值给y，并且要删除i2中¬C(y)
                                                        #以下为格式输出的写入
                                                        re = road(ttt,tvar[0],str((int(sen.index(i1)+1))), str((int(sen.index(i2)+1))), '',1)
                                                        if len(i2)>1 :
                                                            re.ma = str((int(sen.index(i2)+1)))+str(chr(int(j2)+97))
                                                        if len(i1)>1 :
                                                            re.fa = str((int(sen.index(i1)+1)))+str(chr(int(j1)+97))
                                                        #以下将tony赋值给y，删除i2中¬C(y)：通过遍历该元素的所有变量，并与之前“确定”的值比较
                                                        for p in tnum :
                                                            for ff in p.var :
                                                                if ff == ttt:
                                                                    p.var.insert(p.var.index(ff), tvar[0])
                                                                    p.var.remove(ff) 
                                                        #以下为格式输出的写入
                                                        for tr in tnum :    
                                                            if tr.tf ==False :
                                                                tstr = '¬'+tr.name+str(tr.var)
                                                            else :
                                                                tstr = tr.name+str(tr.var) 
                                                            re.w.append(tstr) 
                                                        re.where = len(sen)+1
                                                        #以下通过findsame函数查重，如果归结结果已在子句集中，则不再写入
                                                        if findsame(sen,tnum)==0 :
                                                            rd.append(re)
                                                            sen.append(tnum)
                                                        #以下判断新的子句是否有变量是未知量
                                                        jjjjj = 0
                                                        for ii in tnum :
                                                            for iii in ii.var :
                                                                if iii not in var :
                                                                    jjjjj = 1
                                                        if jjjjj == 0 :
                                                            jug.append(1)
                                                        else :
                                                            jug.append(0)
                                                        #如果归结出空集，结束进程
                                                        if len(tnum)==0:
                                                            return
                                    else : 
                                        if len(i1)==1 :  #在归结时会产生出有多个元素且变量全为已知量的子句，如果归结子句元素单一
                                            #操作与上述判断多个变量近似，便不再赘述
                                            if ((i1[j1].var[0] in var)&(i1[j1].var!=i2[j2].var)&(i2[j2].var[0] not in var))|((i1[j1].var[0] in var)&(i1[j1].var==i2[j2].var)) :  
                                                    tnum = copy.deepcopy(i2)
                                                    re = road(i2[j2].var,i1[j1].var, str((int(sen.index(i1)+1))), str((int(sen.index(i2)+1))), '',1)   
                                                    if len(i1)>1 :
                                                        re.fa = str((int(sen.index(i1)+1)))+str(chr(int(j1)+97))
                                                    if len(i2)>1 :
                                                        re.ma = str((int(sen.index(i2)+1)))+str(chr(int(j2)+97))
                                                    ttt = str(i2[j2].var[0])
                                                    for p in tnum :
                                                        for pp in p.var :
                                                            if (ttt== pp) :
                                                                p.var.insert(p.var.index(ttt), i1[j1].var[0])
                                                                p.var.remove(ttt)
                                                    tnum.pop(j2)
                                                    for tr in tnum :    
                                                        if tr.tf ==False :
                                                            tstr = '¬'+tr.name+str(tr.var)
                                                        else :
                                                            tstr = tr.name+str(tr.var)
                                                        re.w.append(tstr) 
                                                    re.where = len(sen)+1    
                                                    if findsame(sen,tnum)==0 :
                                                        rd.append(re)
                                                        sen.append(tnum)
                                                    jug.append(1)    
                                                    if len(tnum)==0:
                                                        return
                                        else :#将归结子句元素有多个
                                            for aa in i1 :#若将归结子句元素多个，其所有元素的变量必须都为已知量
                                                for bb in i2 :#只有对应元素谓词相同，真值相反且变量值相同才可以归结
                                                    if(aa.var == bb.var)&(aa.name==bb.name)&((((aa.tf==False)&(bb.tf==True))|((aa.tf ==True)&(bb.tf ==False)))) :
                                                        #各种情况归结方式都差不多，便不再赘述
                                                        tnum = copy.deepcopy(i1)
                                                        tnum.pop(i1.index(aa))
                                                        re = road(bb.var,aa.var, str((int(sen.index(i1)+1))), str((int(sen.index(i2)+1))), '',1)   
                                                        if len(i1)>1 :
                                                            re.fa = str((int(sen.index(i1)+1)))+str(chr(int(i1.index(aa))+97))
                                                        if len(i2)>1 :
                                                            re.ma = str((int(sen.index(i2)+1)))+str(chr(int(i2.index(bb))+97))
                                                        for tr in tnum :    
                                                            if tr.tf ==False :
                                                                tstr = '¬'+tr.name+str(tr.var)
                                                            else :
                                                                tstr = tr.name+str(tr.var)
                                                            re.w.append(tstr) 
                                                        re.where = len(sen)+1 
                                                        if findsame(sen,tnum)==0 :
                                                            rd.append(re)
                                                            sen.append(tnum)
                                                        jug.append(1)    
                                                        if len(tnum)==0:
                                                            return
                                                    
    #查重函数
    def findsame(sen,p):
        clock = 0
        for i in sen :  #将输入的子句与子句集的每个子句比较判断
            for j in p :
                if len(i)==len(p) :
                    for k in range(len(i)) :
                        if (i[k].name==p[k].name)&(i[k].var==p[k].var)&(i[k].tf==p[k].tf):
                            clock =1
        return clock
    #提取字符串中的数字
    def getdig(s) :
        p=0
        count = 1
        for i in s[::-1] :
            if i.isdigit() :
                p+=count*int(i)
                count=count*10
        return p
    #找到最终能够归结出空集的路径
    def findrd(it) :
        frd[it-1] = 1
        if it > len(sen)-len(rd) :    
            findrd(getdig(rd[it-len(sen)+len(rd)-1].fa))
            findrd(getdig(rd[it-len(sen)+len(rd)-1].ma))  
    #按格式输出                                                            
    def printrd() :
        ts = ''
        for i in rd :
            if i.bef==i.aft :
                if not i.w :
                    #print(str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+' = '+str(i.w))
                    ts=str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+' = '+str(i.w)
                else :
                    #print(str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+' = '+str(i.w).strip('"').strip("[]"))
                    ts=str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+' = '+str(i.w).strip('"').strip("[]")
            else :
                #print(str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+'('+str(i.bef).strip('[]').strip("'")+'='+str(i.aft).strip('[]').strip("'")+')'+' = '+str(i.w).strip('"').strip("[]")) 
                ts=str(i.where)+" : "+'R['+str(i.fa)+','+str(i.ma)+']'+'('+str(i.bef).strip('[]').strip("'")+'='+str(i.aft).strip('[]').strip("'")+')'+' = '+str(i.w).strip('"').strip("[]")
            outp.append(ts)
    mkdict(putin)
    var = list(set(var))
    try :
        do()
    except Exception as e :
        return
    it=len(sen)
    frd= [0]*len(sen)
    findrd(it)
    printrd()
    return osen,frd,outp