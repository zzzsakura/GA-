import numpy as np
import matplotlib.pyplot as plt
import os
import random

DNA_SIZE = 20
POP_SIZE = 200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 100
X_BOUND = [5, 15]
"""
设置随机种子，以便每次结果相同,保证结果可复现
"""
seed = 2023
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class GA(object):
    def __init__(self, X, DNA, POP, CORSS_R, MU_R, N):
        self.X = X
        self.DNA = DNA
        self.C_R = CORSS_R
        self.M_R = MU_R
        self.N = N
        self.POP = np.random.randint(2, size=(POP, DNA * 2))  # matrix (POP_SIZE, DNA_SIZE)

    def F(self, x):
        return 2 * x + x ** 2

    def plot_2d(self):
        X = np.linspace(*X_BOUND, 10)
        Y = self.F(X)
        plt.xlim(5, 15)
        plt.plot(X, Y)
        plt.pause(3)
        plt.show()

    def get_fitness(self, pop):
        pop = pop
        x = self.translateDNA(pop)
        pred = self.F(x)
        return (pred)  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]

    def translateDNA(self, pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        pop = pop
        x_pop = pop[:, 1::2]  # 奇数列表示X,从索引列1开始，加入了步长2
        x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[
            0]

        return x

    def crossover_and_mutation(self):
        pop = self.POP
        CROSSOVER_RATE = self.C_R
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop

    def mutation(self, child, MUTATION_RATE=0.003):

        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

    def select(self,pop,fitness):  # nature selection wrt pop's fitness
        pop = pop
        fitness = fitness
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                               p=fitness / (fitness.sum()))
        return pop[idx]

    def print_info(self, pop):
        pop = pop
        fitness = self.get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        x = self.translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("x:", (x[max_fitness_index]))

    def start(self):
        global pop
        for _ in range(N_GENERATIONS):  # 迭代N代
            pop = self.POP
            x = self.translateDNA(pop)
            plt.scatter(x, self.F(x), c='black', marker='o')
            plt.show()
            plt.pause(0.1)
            pop = np.array(ga.crossover_and_mutation())
            fitness = ga.get_fitness(pop)
            pop = self.select(pop,fitness)  # 选择生成新的种群
        return pop


if __name__ == "__main__":
    ga = GA(X_BOUND, DNA_SIZE, POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, N_GENERATIONS)
    fig = plt.figure()
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    ga.plot_2d()
    pop = ga.start()
    ga.print_info(pop)
    plt.ioff()
    ga.plot_2d()
