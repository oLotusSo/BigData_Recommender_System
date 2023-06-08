import math
import os
import numpy as np
from operator import itemgetter
import joblib
from tqdm import tqdm

class UserCF_demo():
    def __init__(self):
        self.n_sim_user = 500  # 考虑的相似用户数
        self.n_rec_movie = 6  # 为用户推荐item的数目

        # 训练集与测试集,格式为{用户:{item:评分...}}
        self.trainSet = {}
        self.testSet = {}

        # 用户相似度，格式为{用户:{用户:相似度...}}
        self.user_sim_matrix = {}
        self.movies_count = 455705

        self.OUTPUTFILE = './out.txt'

    def get_dataset(self, filename, testfile):
        """
        构建{用户:{item:评分...}}的训练集与测试集
        :param filename: rating文件路径
        :return:
        """
        with open(filename) as f:
            for line in f:
                list1 = line.split('|')
                userId = int(list1[0])
                itemNum = int(list1[1])
                self.trainSet.setdefault(userId, {})
                for k in range(itemNum):
                    line1 = f.readline()
                    list2 = line1.split()
                    itemId = int(list2[0])
                    rating = int(list2[1])
                    self.trainSet[userId][itemId] = rating 
        print("Loading rating file complete!")
        with open(testfile) as f:
            for line in f:
                list1 = line.split('|')
                userId = int(list1[0])
                itemNum = int(list1[1])
                self.testSet.setdefault(userId, {})
                for k in range(itemNum):
                    line1 = f.readline()
                    itemId = int(line1)
                    self.testSet[userId][itemId] = 0
        print("Loading test file complete!")

    def calc_users_sim(self):
        """
        通过倒排索引将训练集转为{物品:(用户列表),...},再
        利用余弦相似度，计算用户相似度，格式为{用户:{用户:相似度...}}
        :return:
        """
        # 构建倒排索引表
        movie_users = {}
        for user, movies in self.trainSet.items():
            for movie in movies.keys():
                if movie not in movie_users:
                    movie_users.setdefault(movie, set())
                movie_users[movie].add(user)
        print("Bulid inverted index complete!")
        print("It will take about 45 min to build co-matrix.")
        # 构建共现矩阵

        for movie, users in tqdm(movie_users.items()):
            userNum = len(users)
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    # 由于热门item不能差异化地度量人之间的相似性，因此，需要降低热门item的权重
                    self.user_sim_matrix[u][v] += 1 / math.log(1 + userNum)

        print("Bulid co-matrix complete!")
        print("It will take about 15 min to calculate.")
        # 计算用户相似度-余弦相似度=共现item数 / 各自item数的乘积的开方
        for u, related_users in tqdm(self.user_sim_matrix.items()):
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / np.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print("cal users simility complete!")
        # self.save_model()
    
    def save_model(self):
        with open('model.pkl', 'wb') as f:
            joblib.dump(self.user_sim_matrix, f)
        print("Save model complete!")

    def load_model(self):
        with open('model.pkl', 'rb') as f:
            self.user_sim_matrix = joblib.load(f)
        print("Load model complete!")
        
    def recommend(self, user):
        """
        利用用户相似度和相似用户对某个item的加权平均获得目标用户对某个item评价预测
        :return:
        """
        if len(self.user_sim_matrix[user]) > self.n_sim_user:
            K = self.n_sim_user
        else:
            K = len(self.user_sim_matrix[user])
        rank = {}
        # 降序取前K个相似用户
        totalSim = 0
        for v, sim in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:K]:
            # 得到相似用户的rating
            totalSim += sim
            for m, r in self.testSet[user].items():
                rank.setdefault(m, 0)
                for movie, ra  in self.trainSet[v].items():
                    if m == movie:
                        rate = self.trainSet[v][m]
                        rank[m] += sim * float(rate)

        with open(self.OUTPUTFILE, 'a') as file1:
            print("{}|6".format(user), file = file1)
            for item, score in rank.items():
                if totalSim == 0:
                    print("{} {}".format(item, round(score, 3)), file = file1)
                else:
                    print("{} {}".format(item, round(score / totalSim, 3)), file = file1)

    def result(self):
        with open(self.OUTPUTFILE, 'r+', encoding = 'utf-8') as outfile:
            outfile.truncate(0) # 清空文件
        for i, user in enumerate(self.trainSet):
            self.recommend(user)

if __name__ == '__main__':
    rating_file = r'train.txt'
    test_file = r'test.txt'
    userCF = UserCF_demo()
    # 构建训练集与测试集
    userCF.get_dataset(rating_file, test_file)
    # 计算用户相似度
    if os.path.exists('model.pkl'):
        userCF.load_model()
    else:
        userCF.calc_users_sim()
    # 生成最终推荐结果
    userCF.result()