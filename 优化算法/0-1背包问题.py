import numpy as np
from pprint import pprint

# 初始化种群 popsize表示种群中个体个数 n代表基因长度(也就是物品种类数)
def init(popsize, n):
    population = []
    for i in range(popsize):
        pop = ""
        for j in range(n):
            pop += str(np.random.randint(0, 2))
        population.append(pop)
    return population

# 计算各种01情况下的profit和weight
def computeFitness(population, weight, profit):
    total_weight = []
    total_profit = []
    for pop in population:
        weight_temp = 0
        profit_temp = 0
        for index in range(len(pop)):
            if pop[index] == '1':
                weight_temp += int(weight[index])
                profit_temp += int(profit[index])
        total_weight.append(weight_temp)
        total_profit.append(profit_temp)
    return total_weight, total_profit

#自然选择
def roulettewheel(popsize, population, total_profit):
    sum_profit = 0
    p = []
    temp = 0
    for x in total_profit:
        sum_profit += int(x)
    for i in range(len(total_profit)):
        unit = int(total_profit[i]) / sum_profit
        p.append(temp+unit)
        temp += unit

    new_population = []
    while len(new_population) < popsize:
        select_p = np.random.uniform()
        if select_p <= p[0]:
            new_population.append(population[0])
        for index in range(1, len(p)):
            if p[index - 1] < select_p <= p[index]:
                new_population.append(population[index])
    return new_population

# 交叉
def ga_cross(new_population, pcross):
    new = []
    while len(new) < len(new_population):
        if (np.random.uniform() < pcross):
            mother_index = np.random.randint(0, len(new_population))
            father_index = np.random.randint(0, len(new_population))
            threshold = np.random.randint(0, len(new_population[0]))
            if father_index != mother_index:
                temp1 = new_population[father_index][:threshold]
                temp2 = new_population[father_index][threshold:]
                temp3 = new_population[mother_index][:threshold]
                temp4 = new_population[mother_index][threshold:]
                child1 = temp1 + temp4
                child2 = temp2 + temp3
                new.append(child1)
                new.append(child2)
    return new

# 变异
def mutation(population, pm):
    for index in range(len(population)):
        if (np.random.uniform() < pm):
            pos = np.random.randint(0, len(population[0]))
            if population[index][pos] == '0':
                population[index] = population[index][:pos] + '1' + population[index][pos + 1:]
            else:
                population[index] = population[index][:pos] + '0' + population[index][pos + 1:]
    return population

def select(population, weight_limit, total_weight, total_profit):
    w = []
    p = []
    pop = []
    for index in range(len(total_weight)):
        if total_weight[index] < weight_limit:
            w.append(total_weight[index])
            p.append(total_profit[index])
            pop.append(population[index])
    return pop, w, p


if __name__ == '__main__':
    pm = 0.08        # 突变概率
    pc = 0.6        # 交叉概率
    iters = 200      # 迭代次数
    pop_size = 100    # 种群数量
    n = 14          # 物件数量
    weight = [5, 7, 9, 8, 4, 3, 10, 14, 13, 9, 6, 8, 5, 15]     # 物品重量
    profit = [10, 8, 15, 9, 6, 5, 20, 10, 13, 10, 7, 12, 5, 18] # 物品价值
    weight_limit = 75
    population = init(pop_size, n)
    best_pop = []
    best_p = []
    best_w = []
    s_iter = 0
    print("初始为:")
    pprint(population)
    while s_iter < iters:
        print(f"第{s_iter + 1}代")
        w, p = computeFitness(population, weight, profit)
        s_pop, s_w, s_p = select(population, weight_limit, w, p)
        best_index = s_p.index(max(s_p))
        best_pop.append(s_pop[best_index])
        best_p.append(s_p[best_index])
        best_w.append(s_w[best_index])
        # print(f'筛选后的种群{s_pop},筛选后的weight{s_w}, 筛选后的profit{s_p}')
        new_pop = roulettewheel(pop_size, s_pop, s_p)
        # print("选择后的:", new_pop)
        new_pop1 = ga_cross(new_pop, pc)
        # print("交叉后的:", new_pop1)
        population = mutation(new_pop1, pm)
        # print("变异后的:", population)
        s_iter += 1
        # print('-------'*5)
    best_i = best_p.index(max(best_p))
    print(f'实验参数为:变异阈值:{pm},交叉阈值{pc},种群数量{pop_size}')
    print(f'在该实验参数下，总共迭代{iters}次,最优解为{best_pop[best_i]},profit为{best_p[best_i]},weight为{best_w[best_i]}')
