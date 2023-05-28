import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def draw(p1,p2) :
    plt.plot(p2,p1)
    plt.xlabel('study_rate')
    plt.ylabel('correct_rate')
    plt.show()
def run(study_rate) :
    total_word = {}	#所有出现的单词
    train_sentence = []	#训练文本出现语句
    train_set = 'data\\Classification\\train.txt'
    test_set = 'data\\Classification\\test.txt'
    # train_set = 'data\\Classification\\test.txt'
    # test_set = 'data\\Classification\\train.txt'
    loss = []
    totalw2 = []
    epoch = []
    emotion = ["anger" , "disgust" , "fear" , "joy" , "sad" , "surprise"]
    def get_train() :	#获取所有出现单词
        with open(train_set) as f:
            for sentence in f.readlines():	#按行读文件   
                temp = sentence.split(" ")	
                train_sentence.append(temp)	#存入训练文本语句
                for i in range(3 , len(temp)):	#将文本单词存入total_word
                    if(temp[i] not in total_word):#新单词为其设置新的标记
                        total_word[temp[i]] = len(total_word)
    def jugde_which_emotion(out) :#判断文本情感
        res = np.zeros((6,1))
        max = out.max()
        for i in range(6) :	#取输出矩阵中最大值所对应的情感作为文本情感
            if(out[i,0]==max) :
                res[i,0] = 1
        return res
    def sigmoid(out):
        return 1 / (1 + np.exp(-out))
    def get_emotion(pin) :#获取文本情感对应的矩阵
        res = np.zeros((6,1))
        if(pin == "anger") :
            res[0,0] = 1
        elif(pin == "disgust") :
            res[1,0] = 1
        elif(pin == "fear") :
            res[2,0] = 1
        elif(pin == "joy") :
            res[3,0] = 1
        elif(pin == "sad") :
            res[4,0] = 1
        elif(pin == "surprise") :
            res[5,0] = 1
        return res
    def get_stand(out) :    #out数组标准化
        sum = np.sum(out)
        for i in range(len(out)) :
            out[i][0] = out[i][0]/sum
        return out 
    def train(w1,w2,total_word,learning_rate,c) :
        current_right = 0	#预测情感正确数
        for i in range(len(train_sentence)) :	#对学习文本逐句进行训练
            inp = np.zeros((len(total_word),1))
            for w in train_sentence[i][3:] :
                inp[total_word[w],0] = 1
            hide = np.dot(w1,inp)	#根据输入得到隐藏层
            # out = sigmoid(np.dot(w2,hide))
            out = get_stand(sigmoid(np.dot(w2,hide)))	#根据隐藏层得到输出
            # print(out)   
            # out_loss = jugde_which_emotion(out)-get_emotion(train_sentence[i][2])
            # out_delta = out_loss*jugde_which_emotion(out)*(1-jugde_which_emotion(out))
            out_loss = out - get_emotion(train_sentence[i][2])
            out_delta = out_loss*(1-out)
            # out_delta = out_loss*(1-out)*out
            hide_loss = np.dot(w2.T, out_loss)    #损失函数
            hide_delta = hide_loss   #求偏导
            
            w2 -= np.dot(out_delta, hide.T)*learning_rate   #w = w - Learnin g_rate * dw
            w1 -= np.dot(hide_delta, inp.T)*learning_rate
            if (np.sum(out_loss)<0.5) :
                loss.append(np.mean(out_loss))
                c += 1 
                epoch.append(c)
                totalw2.append(np.sum(w2))
            for j in range(6) :
                if jugde_which_emotion(out)[j,0] == 1 :
                    break
            if emotion[j] == train_sentence[i][2] :
                current_right += 1 
            # print('correct : '+ str(current_right) + '  rate : ' + str(current_right / (i+1)))
        return current_right,i+1,c

    def test(w1,w2,total_word) : 
        right_num = 0	#预测正确数
        line = 0	#测试文本句数
        with open(test_set) as f0 :
            with open('result.txt', "w") as f1:
                for sent in f0.readlines() :
                    line += 1	#按行读入预测文本
                    inp = np.zeros((len(total_word),1))
                    words = sent.split(' ')
                    for i in words[3:] :	#获取输入层
                        if i in total_word :
                            inp[total_word[i],0] = 1
                    hide = sigmoid(np.dot(w1,inp))	#按照训练得到权重计算
                    out = sigmoid(np.dot(w2,hide))

                    for i in range(6) :	#判断表达情感
                        if jugde_which_emotion(out)[i,0] == 1 :
                            break
                    if emotion[i] == words[2] :	#判断是否预测成功
                        right_num += 1
                        f1.write('correct ')
                    else :
                        f1.write('wrong ')
                    f1.write(' pridict :'+emotion[i])
                    f1.write(" "+sent+"\n")
        print('correct : '+ str(right_num)+' rate : ' + str(right_num/line))
        return right_num/line
    def draw_loss() :
        plt.scatter(epoch,loss,s=1)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    get_train()
    learning_rate = study_rate
    rd = np.random.RandomState(7) 
    w1 = rd.uniform(0, len(total_word)**-0.5, (100,len(total_word)))
    w2 = rd.uniform(0, 100**-0.5, (6,100))
    # w1 = np.random.normal(0, len(total_word)**-0.5, (100,len(total_word)))
    # w2 = np.random.normal(0, 100**-0.5, (6,100))
    c = 0
    right_num,line,c = train(w1, w2, total_word, learning_rate,c) 
    for i in range(12) :
        right_num,line,c = train( w1, w2, total_word, learning_rate,c)
        i+=1
        print('epoch ' + str(i) + ' correct : '+ str(right_num)+' rate : ' + str(right_num/line))
    outrate=test(w1, w2, total_word)

    draw_loss()
    return (outrate)
if __name__ == '__main__':
    corret_rate = []
    study_rate = [0.007,0.01,0.05,0.1,0.105,0.15,0.115,0.1225,0.125,0.2]
    for i in range(10) :
        rate = run(study_rate[i])
        corret_rate.append(rate)
    draw(corret_rate,study_rate)
    # run(0.1)