"""
Solve using the modified MACO

Version description, optimization for supporting mega frame core tube structures
Structure brief information: 100 floors, with a height of 5 meters
The optimization parameters are 19, respectively
MC1，MC2,MC3,MC4,MC5
W1,W2,W3,W4,W5
MB1,MB2,MB3,MB4,MB5
h,w,t,OT_ FG
12 constraint conditions, respectively
r_ C1,r_ C2,r_ C3,r_ C4,r_ C5
r_ W1,r_ W2,r_ W3,r_ W4,r_ W5
r_ BM,storydrift_ max
"""
import random
import numpy as np
from sklearn.externals import joblib
import time
import matplotlib.pyplot as plt
import pandas as pd

dim = 19
size= 200
iter_num = 50
v_conc = 1

POP_SIZE = size
DNA_SIZE = dim
N_GENERATIONS = iter_num
steelPrice = 5000
concretePrice = 500
limit_C = 0.6
limit_W = 0.5
limit_BM = 1
limit_story_max = 0.002 #1/500
ek = (235/345)**0.5
A_MB=230400


W1_lb = 1000;W1_ub = 4000;W2_lb=0.7;W2_ub=1.0;W3_lb = 0.7;W3_ub = 1.0;W4_lb=0.7;W4_ub=1.0;W5_lb=0.7;W5_ub=1.0
MC1_lb = 4000;MC1_ub = 8000;MC2_lb=0.7;MC2_ub=1.0;MC3_lb = 0.7;MC3_ub = 1.0;MC4_lb=0.7;MC4_ub=1.0;MC5_lb=0.7;MC5_ub=1.0
MB1_lb = 200;MB1_ub = 5500;MB2_lb = 200;MB2_ub = 5500;MB3_lb = 200;MB3_ub = 5500
MB4_lb = 200;MB4_ub = 5500;MB5_lb = 200;MB5_ub = 5500;OT_FG_lb=200;OT_FG_ub = 5500
t_lb = 20;t_ub = 50;

r_W1_predmodel = joblib.load('r_W1_predmodel.pkl')
r_W2_predmodel = joblib.load('r_W2_predmodel.pkl')
r_W3_predmodel = joblib.load('r_W3_predmodel.pkl')
r_W4_predmodel = joblib.load('r_W4_predmodel.pkl')
r_W5_predmodel = joblib.load('r_W5_predmodel.pkl')

r_C1_predmodel = joblib.load('r_C1_predmodel.pkl')
r_C2_predmodel = joblib.load('r_C2_predmodel.pkl')
r_C3_predmodel = joblib.load('r_C3_predmodel.pkl')
r_C4_predmodel = joblib.load('r_C4_predmodel.pkl')
r_C5_predmodel = joblib.load('r_C5_predmodel.pkl')

storydriftmax_predmodel = joblib.load('storydriftmax_pred_model.pkl')
r_BM_predmodel = joblib.load('r_BM_predmodel.pkl')

#Optimization parameters: giant columns (MC1-MC5), shear walls (W1-W5), steel beams (h, w, web_t), supports (MB1-MB5)

#Constraint conditions: axial compression ratio of giant columns (rC1-rC5), axial compression ratio of shear walls (rCW1-rCW5), stress ratio of steel beams (rBM), interlayer displacement angle (Story_max)

#Optimization target Construction money

#Working condition EQX
#The following is for the convenience of modifying the code, using genetic algorithm as the variable name

def translateDNA(pop): #Decoding
    realchangers = np.zeros((POP_SIZE, DNA_SIZE))
    for ii in range(POP_SIZE):
        MC1 = pop[ii, 0]; MC2 = int(pop[ii, 1]*MC1);MC3 = int(pop[ii, 2]*MC2);MC4 = int(pop[ii, 3]*MC3);MC5 = int(pop[ii, 4]*MC4);
        W1  = pop[ii, 5]; W2 = int(pop[ii, 6]*W1);W3 = int(pop[ii, 7]*W2);W4 = int(pop[ii, 8]*W3);W5 = int(pop[ii, 9]*W4);
        MB1 = pop[ii, 10]/1000; MB2 = pop[ii, 11]/1000;MB3 = pop[ii, 12]/1000;MB4 = pop[ii, 13]/1000;MB5 = pop[ii, 14]/1000
        h = pop[ii, 15]; w = pop[ii, 16]; t = pop[ii, 17];OT_FG= pop[ii, 18]/1000;
        realchangers[ii, :] = MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG
        print('Parameters of model','MC1',MC1,'MC2',MC2,'MC3',MC3,'MC4',MC4, 'MC5',MC5, \
              'W1',W1,'W2',W2,'W3',W3,'W4',W4, 'W5',W5, \
              'MB1',MB1,'MB2',MB2,'MB3',MB3,'MB4',MB4, 'MB5',MB5, \
              'h',h, 'w',w, 't',t,'OT_FG',OT_FG)

    return realchangers


def F(num,realchangers): #Solve
    r_C1 = np.zeros((POP_SIZE, 1)); r_C2 = np.zeros((POP_SIZE, 1));r_C3 = np.zeros((POP_SIZE, 1));
    r_C4 = np.zeros((POP_SIZE, 1));r_C5 = np.zeros((POP_SIZE, 1));
    r_W1 = np.zeros((POP_SIZE, 1)); r_W2 = np.zeros((POP_SIZE, 1));r_W3 = np.zeros((POP_SIZE, 1));
    r_W4 = np.zeros((POP_SIZE, 1));r_W5 = np.zeros((POP_SIZE, 1));
    r_BM = np.zeros((POP_SIZE, 1)); story_max = np.zeros((POP_SIZE, 1));
    money_consc = np.zeros((POP_SIZE, 1))

    for i in range(POP_SIZE):
        MC1 = realchangers[i, 0]; MC2 = realchangers[i, 1];MC3 = realchangers[i, 2];
        MC4 = realchangers[i, 3];MC5 = realchangers[i, 4];
        W1 = realchangers[i, 5]; W2 = realchangers[i, 6];W3 = realchangers[i, 7];
        W4 = realchangers[i, 8];W5 = realchangers[i, 9];
        MB1 = realchangers[i, 10]; MB2 = realchangers[i, 11];MB3 = realchangers[i, 12];
        MB4 = realchangers[i, 13];MB5 = realchangers[i, 14];
        h = realchangers[i, 15]; w = realchangers[i, 16]
        t = realchangers[i, 17];web_t = t;OT_FG = realchangers[i, 18]
        A_OT_FG=131200
        print('第',str(num+1),'代','第',str(i+1),'个体进行计算...')
        # r_C1[i], r_C3[i], r_W1[i], r_W3[i], r_BM[i],story_max[i],\
        #  = multi_runetabs.run(num1,MC1, MC3, W1, W3, MB1, MB3, h,w, t)
        a = t*(h-2*t)**3+2*(w*t**3/12+w*t*(h/2-t/2)**2)
        P = np.array([MC1,MC2,MC3,MC4,MC5,W1,W2,W3,W4,W5,230400*MB1,230400*MB2,230400*MB3,230400*MB4,230400*MB5,h,w,t,\
                      web_t,OT_FG*A_OT_FG,3.8*MC1**4/12,\
                      3.8*MC2**4/12,3.8*MC3**4/12,3.8*MC4**4/12,3.8*MC5**4/12,3.8*MC1**2,3.8*MC2**2,\
                      3.8*MC3**2,3.8*MC4**2,3.8*MC5**2,2*W1,2*W2,2*W3,2*W4,2*W5,7.8*(h*w-(w-t))])
        r_C1[i] = r_C1_predmodel.predict(P.reshape(1,-1))
        r_C2[i] = r_C2_predmodel.predict(P.reshape(1,-1))
        r_C3[i] = r_C3_predmodel.predict(P.reshape(1,-1))
        r_C4[i] = r_C4_predmodel.predict(P.reshape(1,-1))
        r_C5[i] = r_C5_predmodel.predict(P.reshape(1,-1))

        r_W1[i] = r_W1_predmodel.predict(P.reshape(1,-1))
        r_W2[i] = r_W2_predmodel.predict(P.reshape(1,-1))
        r_W3[i] = r_W3_predmodel.predict(P.reshape(1,-1))
        r_W4[i] = r_W4_predmodel.predict(P.reshape(1,-1))
        r_W5[i] = r_W5_predmodel.predict(P.reshape(1,-1))

        r_BM[i] = r_BM_predmodel.predict(P.reshape(1,-1))
        story_max[i] = storydriftmax_predmodel.predict(P.reshape(1,-1))
        money_consc[i]=money(MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG)
        print('柱轴压比C1:',r_C1[i],r_C2[i],r_C3[i],r_C4[i],r_C5[i])
        print('墙轴压比W1:',r_W1[i],r_W2[i],r_W3[i],r_W4[i],r_W5[i])
        print('梁应力比r_BM:',r_BM[i],'最大层间位移角:',story_max[i])
        print('造价为',money_consc[i])
    return r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc

#MC1, MC3 side length mm

#W1, W3 thickness mm

#MB1, MB3 area mm2

#H, w, t length mm
def money(MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG):
    A_BM = h * w - (h - 2 * t) * (w - t)
    A_B1_PT_FG = 131200;A_B1_OT_XG= 230400;A_B1_OT_FG= 131200;A_B1_PT_XG= 230400
    Steel_Money = (steelPrice * 7.85 / (10 ** 9)) * (A_BM * 13180 * 170 + (MB1 +\
                 MB2 + MB3 + MB4 + MB5) *0.001 *A_MB* (57240 * 2) +A_B1_OT_XG * (13180 * 20 \
                 + 10500 * 20 + 9000 * 10) + OT_FG*A_B1_OT_FG * (16540 * 20 + 14500 *\
                 20)+ A_B1_PT_FG * 17280 * 40 + A_B1_PT_XG * 14090 * 40)
    Conc_Money = (concretePrice / (10 ** 9)) * ((MC1 ** 2 + MC2 ** 2 + MC3 **2+\
                  MC4 ** 2 + MC5 ** 2) * (5000 * 36 + 10000 * 2) + (W1 + W2 + \
                  W3 + W4 + W5) * ((52.5 + 10.5 + 4.5 + 4.5 + 4.5 + 10.5 + 52.5)\
                  * 18 * 10 ** 6 + (105 + 25.5 + 4.5 + 4.5 +\
                  4.5 + 25.5 + 105) * 10 ** 6))
    money_sum = Conc_Money +Steel_Money
    return money_sum

def fit_app(x): #Auxiliary fitness RELU combined with sigmoid, plus all the maximum values
    fit_added = np.exp(x)
    return fit_added

#Fitness

#Both meet 0, and the interlayer displacement angle 
#meets the requirements of the bottom layer but not the middle layer
def get_fitness(r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc):
    #Manufacturing step function


    fitness=np.zeros((POP_SIZE,1))
    for i in range(POP_SIZE):

        rr_C1 = r_C1[i];rr_C2 = r_C2[i];rr_C3 = r_C3[i];rr_C4 = r_C4[i];rr_C5 = r_C5[i];
        rr_W1 = r_W1[i];rr_W2 = r_W2[i];rr_W3 = r_W3[i];rr_W4 = r_W4[i];rr_W5 = r_W5[i];
        rr_BM = r_BM[i];rstory_max = story_max[i];rmoney_consc = money_consc[i];

        sumlimt= np.array([rr_C1 / limit_C, rr_C2 / limit_C,rr_C3 / limit_C,rr_C4 / limit_C,rr_C5 / limit_C,\
                           rr_W1 / limit_W, rr_W2 / limit_W,rr_W3 / limit_W,rr_W4 / limit_W,rr_W5 / limit_W,\
                           rr_BM / limit_BM, rstory_max / limit_story_max])
        m=np.sum(sumlimt>1.0)

        if rstory_max <= limit_story_max:
            a_smax =1
        else:
            a_smax = fit_app(rstory_max/limit_story_max)+m
        #Axial compression ratio of columns


        # C1
        if rr_C1 <=limit_C:
            a_C1 = 1
        else:
            a_C1 =fit_app(rr_C1/limit_C)+m
        # C2
        if rr_C2 <=limit_C:
            a_C2 = 1
        else:
            a_C2 =fit_app(rr_C2/limit_C)+m
        # C3
        if rr_C3 <=limit_C:
            a_C3 = 1
        else:
            a_C3 =fit_app(rr_C3/limit_C)+m
        # C4
        if rr_C4 <=limit_C:
            a_C4 = 1
        else:
            a_C4 =fit_app(rr_C4/limit_C)+m
        # C5
        if rr_C5 <=limit_C:
            a_C5=1
        else:
            a_C5=fit_app(rr_C5/limit_C)+m
        # W1
        if rr_W1 <=limit_W:
            a_W1=1
        else:
            a_W1=fit_app(rr_W1/limit_W)+m
        # W2
        if rr_W2 <= limit_W:
            a_W2 = 1
        else:
            a_W2 =fit_app(rr_W2/limit_W)+m
        # W3
        if rr_W3 <= limit_W:
            a_W3 = 1
        else:
            a_W3 =fit_app(rr_W3/limit_W)+m
        # W4
        if rr_W4 <= limit_W:
            a_W4 = 1
        else:
            a_W4 = fit_app(rr_W4 / limit_W) + m
        # W5
        if rr_W5 <= limit_W:
            a_W5 = 1
        else:
            a_W5 = fit_app(rr_W5 / limit_W) + m
        #BM
        if rr_BM <=limit_BM:
            a_BM=1
        else:
            a_BM=fit_app(rr_BM/limit_BM)+m

        fitness[i] = money_consc[i,0]*a_C1*a_C2*a_C3*a_C4*a_C5*a_BM*a_W1*a_W2*a_W3*a_W4*a_W5*a_smax
    fitness = 1/ fitness

    return fitness

def getpoprange(pop):
    poprange = np.zeros([POP_SIZE, 2*DNA_SIZE])

    for i in range(POP_SIZE):
        h_lb=300;h_ub=min(int(72*ek*float(pop[i,-2])),1300)+2*float(pop[i,-2])
        w_lb=100;w_ub=2*min(int(11 * ek * float(pop[i,-2])),int(float(pop[i,-4])/1.5), 300)+float(pop[i,-2])
        lst = [MC1_lb,MC1_ub,MC2_lb,MC2_ub,MC3_lb,MC3_ub,MC4_lb,MC4_ub,MC5_lb,MC5_ub,\
               W1_lb,W1_ub,W2_lb,W2_ub,W3_lb,W3_ub,W4_lb,W4_ub,W5_lb,W5_ub,\
               MB1_lb,MB1_ub,MB2_lb,MB2_ub,MB3_lb,MB3_ub,MB4_lb,MB4_ub,MB5_lb,MB5_ub,\
               h_lb,h_ub,w_lb,w_ub,t_lb,t_ub,OT_FG_lb,OT_FG_ub]
        poprange[i,:] =pd.array(lst)
    return poprange

def getspeedrange(pop):
    # The speed range limit
    # W1_lb = 5000;W1_ub = 8000;W3_lb = 0.3;W3_ub = 0.8
    # MC1_lb = 4000;MC1_ub = 12000;MC3_lb = 0.49;MC3_ub = 0.8;MB1_lb = 200;
    # MB1_ub = 5500;MB3_lb = 200;MB3_ub = 5500;t_lb = 20;t_ub = 50;
    # P_W1 = random.randrange(5000, 8000, 200); P_W3=random.randrange(300,800,50)/1000
    # P_MC1 = random.randrange(4000,12000,200); P_MC3=random.randrange(490,800,30)/1000
    # P_MB1=random.randrange(200,5500,50); P_MB3=random.randrange(200,5500,50)
    # P_t = random.randrange(20,50,2);
    # P_tw = P_t; h0 = random.randrange(300,min(int(72*ek*P_tw),1300),20)
    # b0 = random.randrange(100, min(int(11 * ek * P_t),int(h0/1.5), 300), 20)
    speedrange = np.zeros([POP_SIZE,DNA_SIZE])
    for i in range(POP_SIZE):
        h_lb=300;h_ub=min(int(72*ek*float(pop[i,-2])),1300)+2*float(pop[i,-2])
        w_lb=100;w_ub=2*min(int(11 * ek * float(pop[i,-2])),int(float(pop[i,-4])/1.5), 300)+float(pop[i,-2])
        speedrange[i] = (v_conc*(MC1_ub-MC1_lb),v_conc*(MC2_ub-MC2_lb),v_conc*(MC3_ub-MC3_lb),v_conc*(MC4_ub-MC4_lb),v_conc*(MC5_ub-MC5_lb),\
                         v_conc*(W1_ub-W1_lb), v_conc*(W2_ub-W2_lb), v_conc*(W3_ub-W3_lb),v_conc*(W4_ub-W4_lb),v_conc*(W5_ub-W5_lb),\
                         v_conc*(MB1_ub-MB1_lb),v_conc*(MB2_ub-MB2_lb), v_conc*(MB3_ub-MB3_lb), v_conc*(MB4_ub-MB4_lb),v_conc*(MB5_ub-MB5_lb),\
                         v_conc*(h_ub-h_lb),v_conc*(w_ub-w_lb),v_conc*(t_ub-t_lb),v_conc*(OT_FG_ub-OT_FG_lb))
    return speedrange

def updatebest(pop,fitness,gbestpop,gbestfitness,money_consc):
    global rgbestmoney
    if fitness.max() > gbestfitness:
        gbestfitness = fitness.max()
        rgbestmoney = money_consc[fitness.argmax()]
        gbestpop = pop[fitness.argmax()].copy()

    return gbestpop,gbestfitness, rgbestmoney

def calcu_actprob(t_max,t):

    for i in range(POP_SIZE):
        pi[i] = (t_max - t[i]) / t_max

    return pi

def update_t(fitness,t):
    for i in range(POP_SIZE):
        t[i] = (1 - rou) * t[i] + Q * fitness[i]
    t_max = np.max(t)
    return t_max,t

#Main function
start_time = time.time()
pop = np.zeros([POP_SIZE, DNA_SIZE])
t = np.zeros([POP_SIZE, 1])
gbestmoney = np.zeros([N_GENERATIONS,1])
result = np.zeros(N_GENERATIONS)
global rgbestmoney
rgbestmoney = 0
gbestfitness = 0
tmax=0
gbestpop = np.zeros((1, DNA_SIZE))
rou = 0.8  #Volatility coefficient of pheromone


#The total amount of information released
#the total amount of information released by ants in one work cycle

Q = 1  
pi = np.zeros(POP_SIZE)  #Probability table, storing the transition probability of each individual

for i in range(POP_SIZE):
    P_W1 = random.randrange(1000, 4000, 200); P_W2=random.randrange(700,1000,50)/1000;
    P_W3 = random.randrange(700, 1000, 50) / 1000;P_W4=random.randrange(700,1000,50)/1000
    P_W5 = random.randrange(700, 1000, 50) / 1000
    P_MC1 = random.randrange(4000,12000,200); P_MC2=random.randrange(700,1000,50)/1000;
    P_MC3 = random.randrange(700,1000,50) / 1000;P_MC4=random.randrange(700,1000,50)/1000
    P_MC5 = random.randrange(700,1000,50) / 1000
    P_MB1=random.randrange(200,5500,50); P_MB2=random.randrange(200,5500,50);
    P_MB3 = random.randrange(200, 5500, 50);P_MB4=random.randrange(200,5500,50);
    P_MB5 = random.randrange(200, 5500, 50)
    P_OT_FG = random.randrange(200, 5500, 50)
    P_t = random.randrange(20,50,2);
    P_tw = P_t; h0 = random.randrange(300,min(int(72*ek*P_tw),1300),20)
    b0 = random.randrange(100, min(int(11 * ek * P_t),int(h0/1.5), 300), 20)
    P_web_t=P_tw; P_w=2*b0+P_web_t; P_h=h0+2*P_t
    data = (P_MC1, P_MC2, P_MC3, P_MC4, P_MC5, P_W1, P_W2, P_W3, P_W4, P_W5,\
            P_MB1, P_MB2, P_MB3, P_MB4, P_MB5, P_h, P_w, P_t,P_OT_FG)
    pop[i,:] =data

gbestfitness = 0
gbestpop = np.zeros((1, DNA_SIZE))
pbestfitness= np.zeros((POP_SIZE, 1))
pbestpop = np.zeros((POP_SIZE, DNA_SIZE))
v = np.zeros((POP_SIZE, DNA_SIZE))
result = np.zeros((N_GENERATIONS,1))
resultpop = np.zeros((N_GENERATIONS,DNA_SIZE))
for _ in range(N_GENERATIONS):
    num = _ + 1
    realchangers = translateDNA(pop)
    r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc = F(num, realchangers)
    fitness = get_fitness(r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc)
    # Update optimal location, gbestfitness，gbestpop
    if num == 1:
        # Set initial fitness


        gbestfitness = fitness.max()
        gbestmoney[_] = money_consc[fitness.argmax(), 0]
        gbestpop = pop[fitness.argmax()].copy()
    else:
        # Comparison between non beginners


        gbestpop, gbestfitness, gbestmoney[_] = updatebest(pop, fitness, gbestpop, gbestfitness, money_consc)

    result[_] = 1/gbestfitness
    resultpop[_] = gbestpop
    print('Most fitted pop now:', pop[np.argmax(fitness),:])
    print('计算结果为：','\nr_C：',r_C1[np.argmax(fitness)],r_C2[np.argmax(fitness)],r_C3[np.argmax(fitness)],\
          r_C4[np.argmax(fitness)],r_C5[np.argmax(fitness)],'\nr_W：',r_W1[np.argmax(fitness)],r_W2[np.argmax(fitness)],\
          r_W3[np.argmax(fitness)],r_W4[np.argmax(fitness)],r_W5[np.argmax(fitness)],'\nr_BM：',r_BM[np.argmax(fitness)],\
          '\nstory_max：',story_max[np.argmax(fitness)],'\nmoney：',money_consc[np.argmax(fitness)])

    # Update pheromone


    if num == 1:
        t = fitness
        t_max = fitness.max()
    else:
        t_max, t = update_t(fitness, t)

    # Calculate the probability of action


    pi = calcu_actprob(t_max, t)

    # Expand the scope of the ant colony and update the ant position 
	#(global search and local search) based on the probability of ant colony action


    speedrange = getspeedrange(pop)
    # lamda = 1 / num  # Lamda decreases as algebra increases, used for local search


    lamda = np.array([200,0.01,0.01,0.01,0.01,\
    0.01,0.01,0.01,0.01,0.01,\
    0.01,0.01,0.01,0.01,0.01,\
    20,10,1,50])
    for i in range(POP_SIZE):
        for j in range(DNA_SIZE):
            if pi[i] < np.random.uniform(0, 0.6):  #Conduct local search

                pop[i][j] = pop[i][j] + random.randint(0,2) * lamda[j] #Local search for building modulus


            else:  # Conduct a global search
                pop[i][j] = pop[i][j] + np.random.uniform(-1, 1) * float(speedrange[i, j])

    # Restrict ant colony location
    poprange = getpoprange(pop)
    # Limit position range
    for j in range(POP_SIZE):
        for jj in range(DNA_SIZE):
            if pop[j, jj] < poprange[j, 2 * jj]:
                pop[j, jj] = poprange[j, 2 * jj]
            elif pop[j, jj] > poprange[j, 2 * jj + 1]:
                pop[j, jj] = poprange[j, 2 * jj + 1]

plt.plot(list(range(N_GENERATIONS)), list(gbestmoney))
print(gbestpop)

allresult = np.concatenate((result,resultpop),axis=1)
#You can fill in the storage location according to your needs 
savepath = 'allresult3.xlsx'
allresult = pd.DataFrame(allresult)
writer = pd.ExcelWriter(savepath)
allresult.to_excel(writer,'Sheet2')
writer.save()
plt.title("Figure1")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
end_time = time.time()
print('Time:', end_time - start_time)
plt.show()
