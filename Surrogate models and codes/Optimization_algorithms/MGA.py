"""
Solve using the modified GA

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

N_GENERATIONS = 50
POP_SIZE = 4
DNA_SIZE = 19
MUTATION_RATE_all=0.30
CROSS_RATE_Sat = 1
CROSS_RATE_UNSat = 0.3
steelPrice = 5000
concretePrice = 500

MUA_all = np.zeros((DNA_SIZE))
for i in range(DNA_SIZE):
    MUA_all[i] =float(1/DNA_SIZE)
MUA_C =MUA_C1 = MUA_W1 =MUA_W=MUA_CW = MUA_CWB =MUA_all
#MUA_C
for i in range(5):
    MUA_C[i]=1.3/DNA_SIZE
    for i in range(DNA_SIZE-5):
        MUA_C[i+5]=(1-5*MUA_C[0])/(DNA_SIZE-5)
#MUA_C1
MUA_C1[0]=2/DNA_SIZE
for i in range(DNA_SIZE-1):
    MUA_C1[i+1]=(1-MUA_C1[0])/(DNA_SIZE-1)
#MUA_W
for i in range(5):
    MUA_W[i+5]=1.3/DNA_SIZE
    temp=(1-5*MUA_W[5])/(DNA_SIZE-5)
    for i in range(5):
        MUA_W[i] = temp
    for i in range(DNA_SIZE-10):
        MUA_W[i+10]=temp

#MUA_W1
MUA_W1[5]=2/DNA_SIZE
for i in range(DNA_SIZE):
    MUA_W1[i]=(1-MUA_W1[5])/(DNA_SIZE-1)
MUA_W1[5]=2/DNA_SIZE

#MUA_CW
tempcw = 1.3/DNA_SIZE
for i in range(10):
    MUA_CW[i]=tempcw
for i in range(DNA_SIZE-10):
    MUA_CW[i+10]=(1-10*tempcw)/(DNA_SIZE-10)


#MUA_CMB
tempcwb = 1.2/DNA_SIZE
for i in range(15):
    MUA_CWB[i]=tempcwb
for i in range(DNA_SIZE-15):
    MUA_CWB[i+15]=(1-15*tempcwb)/(DNA_SIZE-15)

limit_C = 0.6
limit_W = 0.5
limit_BM = 1
limit_story_max = 0.002 #1/500
ek = (235/345)**0.5
A_MB=230400

def translateDNA(pop): 
    realchangers = np.zeros((POP_SIZE, DNA_SIZE))
    for ii in range(POP_SIZE):
        MC1 = pop[ii, 0]; MC2 = int(pop[ii, 1]*MC1);MC3 = int(pop[ii, 2]*MC2);MC4 = int(pop[ii, 3]*MC3);MC5 = int(pop[ii, 4]*MC4);
        W1  = pop[ii, 5]; W2 = int(pop[ii, 6]*W1);W3 = int(pop[ii, 7]*W2);W4 = int(pop[ii, 8]*W3);W5 = int(pop[ii, 9]*W4);
        MB1 = pop[ii, 10]/1000; MB2 = pop[ii, 11]/1000;MB3 = pop[ii, 12]/1000;MB4 = pop[ii, 13]/1000;MB5 = pop[ii, 14]/1000
        h = pop[ii, 15]; w = pop[ii, 16]; t = pop[ii, 17];OT_FG= pop[ii, 18]/1000;
        realchangers[ii, :] = MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG
        print('Parameters of models','MC1',MC1,'MC2',MC2,'MC3',MC3,'MC4',MC4, 'MC5',MC5, \
              'W1',W1,'W2',W2,'W3',W3,'W4',W4, 'W5',W5, \
              'MB1',MB1,'MB2',MB2,'MB3',MB3,'MB4',MB4, 'MB5',MB5, \
              'h',h, 'w',w, 't',t,'OT_FG',OT_FG)

    return realchangers



def F(num,realchangers): 
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


def money(MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG):
    A_BM = h * w - (h - 2 * t) * (w - t)
    A_B1_PT_FG = 131200;A_B1_OT_XG= 230400;A_B1_OT_FG= 131200;A_B1_PT_XG= 230400
    Steel_Money = (steelPrice * 7.85 / (10 ** 9)) * (A_BM * 13180 * 170 + (MB1 +\
                 MB2 + MB3 + MB4 + MB5)*0.001 *A_MB* (57240 * 2) +A_B1_OT_XG * (13180 * 20 \
                 + 10500 * 20 + 9000 * 10) + OT_FG*A_B1_OT_FG * (16540 * 20 + 14500 *\
                 20)+ A_B1_PT_FG * 17280 * 40 + A_B1_PT_XG * 14090 * 40)
    Conc_Money = (concretePrice / (10 ** 9)) * ((MC1 ** 2 + MC2 ** 2 + MC3 **2+\
                  MC4 ** 2 + MC5 ** 2) * (5000 * 36 + 10000 * 2) + (W1 + W2 + \
                  W3 + W4 + W5) * ((52.5 + 10.5 + 4.5 + 4.5 + 4.5 + 10.5 + 52.5)\
                  * 18 * 10 ** 6 + (105 + 25.5 + 4.5 + 4.5 +\
                  4.5 + 25.5 + 105) * 10 ** 6))
    money_sum = Conc_Money +Steel_Money
    return money_sum

def fit_app(x): 
    fit_added = np.exp(x)
    return fit_added


def get_fitness(r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc):
    
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

        print(fitness[i])
    return fitness


def select(pop, r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc, fitness):


    fitness = (fitness - np.min(fitness))/(np.max(money_consc)-np.min(money_consc))+1*np.exp(-3)
    fitness = 1/fitness
    p1 = fitness / fitness.sum()
    p2 = []
    for i in  range(POP_SIZE):
        p2.append(float(p1[i]))
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=p2)
    return pop[idx]


def crossover(pop,r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc,fitness):
    #MC1, MC2,MC3,MC4,MC5, W1, W2,W3,W4,W5, MB1, MB2,MB3,MB4,MB5, h, w, t, OT_FG
    for i in range (0,POP_SIZE-1):
        rstory_max=story_max[i,0]
        if rstory_max > limit_story_max:
            if random.random() < CROSS_RATE_Sat:
                d = random.randint(0, DNA_SIZE - 15)
                e = random.randint(DNA_SIZE - 14, DNA_SIZE - 10)
                f = random.randint(DNA_SIZE - 9, DNA_SIZE - 5)
                # while True:
                #     e = random.randint(DNA_SIZE-12, DNA_SIZE - 5)
                #     if e > d+1:
                #         break
                j = i+1
                poptemp1 = pop[j,d]
                poptemp2 = pop[j, e]
                poptemp3 = pop[j, f]
                pop[j, d] = pop[i, d].copy()
                pop[j, e] = pop[i, e].copy()
                pop[j, f] = pop[i, f].copy()
                pop[i, d] = poptemp1
                pop[i, e] = poptemp2
                pop[i, f] = poptemp3
        else:
            if random.random() < CROSS_RATE_UNSat: 
                d = random.randint(0, DNA_SIZE - 15)
                e = random.randint(DNA_SIZE - 14, DNA_SIZE - 10)
                f = random.randint(DNA_SIZE - 9, DNA_SIZE - 5)
                j = i + 1
                poptemp1 = pop[j, d]
                poptemp2 = pop[j, e]
                poptemp3 = pop[j, f]
                pop[j, d] = pop[i, d].copy()
                pop[j, e] = pop[i, e].copy()
                pop[j, f] = pop[i, f].copy()
                pop[i, d] = poptemp1
                pop[i, e] = poptemp2
                pop[i, f] = poptemp3

    return pop


def mutate(pop, r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc,fitness):

    for i in range(POP_SIZE):
        rr_C1 = r_C1[i];
        rr_C2 = r_C2[i];
        rr_C3 = r_C3[i];
        rr_C4 = r_C4[i];
        rr_C5 = r_C5[i];
        rr_W1 = r_W1[i];
        rr_W2 = r_W2[i];
        rr_W3 = r_W3[i];
        rr_W4 = r_W4[i];
        rr_W5 = r_W5[i];
        rr_BM = r_BM[i];rstory_max = story_max[i]
        P=MUA_all
        if rr_BM>limit_BM:
            P=MUA_BM
        else:
            if min(rr_C1,rr_C2,rr_C3,rr_C4,rr_C5)>limit_C and min(rr_W1,rr_W2,rr_W3,rr_W4,rr_W5)>limit_W:
                P=MUA_CW
            else:
                if rr_C1>limit_C or rr_C2>limit_C or rr_C3>limit_C or rr_C4>limit_C or rr_C5>limit_C:
                    P=MUA_C
                    if rr_C1>limit_C:
                        P=MUA_C1
                else:
                    if rr_W1>limit_W or rr_W2>limit_W or rr_W3>limit_W or rr_W4>limit_W or rr_W5>limit_W:
                        if rr_W1>limit_W:
                            P=MUA_W1
                        else:
                            P = MUA_W
                    else:
                        if rstory_max<=limit_story_max:
                            P=MUA_all
                        elif rstory_max>limit_story_max:
                            P=MUA_CWB


        M_W1 = random.randrange(1000, 4000, 200)
        M_W2 = random.randrange(490,1000, 50) / 1000
        M_W3 = random.randrange(490, 1000, 50) / 1000
        M_W4 = random.randrange(490, 1000, 50) / 1000
        M_W5 = random.randrange(490, 1000, 50) / 1000

        M_MC1 = random.randrange(4000, 12000, 200)
        M_MC2 = random.randrange(490,1000, 50) / 1000
        M_MC3 = random.randrange(490,1000, 50) / 1000
        M_MC4 = random.randrange(490,1000, 50) / 1000
        M_MC5 = random.randrange(490,1000, 50) / 1000

        M_MB1 = random.randrange(100, 5500, 50)
        M_MB2 = random.randrange(100, 5500, 50)
        M_MB3 = random.randrange(100, 5500, 50)
        M_MB4 = random.randrange(100, 5500, 50)
        M_MB5 = random.randrange(100, 5500, 50)

        OT_FG = random.randrange(100, 5500, 50)

        M_t = random.randrange(20, 50, 2)
        M_tw = M_t
        M_h0 = random.randrange(300,min(int(72*ek*P_tw),1700),20)
        M_b0 = random.randrange(100, min(int(11 * ek * P_t), int(h0 / 1.5), 300), 20)
        M_web_t = M_tw
        M_w = 2 * M_b0 + M_web_t
        M_h = M_h0 + 2 * M_t

        sumlimt= np.array([rr_C1 / limit_C, rr_C2 / limit_C,rr_C3 / limit_C,rr_C4 / limit_C,rr_C5 / limit_C,\
                           rr_W1 / limit_W, rr_W2 / limit_W,rr_W3 / limit_W,rr_W4 / limit_W,rr_W5 / limit_W,\
                           rr_BM / limit_BM, rstory_max / limit_story_max])
        m=np.sum(sumlimt>1.0)
        P = MUA_all
        if m == 0 :
            MUTATION_RATE = MUTATION_RATE_all
        else:
            MUTATION_RATE = 1

        if np.random.rand()<MUTATION_RATE:
            mutate_point = np.random.choice(np.arange(DNA_SIZE),1, p=P)
            if mutate_point == DNA_SIZE-3 or mutate_point == DNA_SIZE-4:
                M_t=pop[i,-2]
                M_b0 = random.randrange(100,min(int(11*ek*P_t),400),20)
                M_tw = M_t;
                M_h0 = random.randrange(300,min(int(72*ek*P_tw),1300),20)
                M_web_t = M_tw;
                M_w = 2 * M_b0 + M_web_t;
                M_h = M_h0 + 2 * M_t
            elif mutate_point == DNA_SIZE-1:
                while pop[i,-4]>=((72*ek+2)*M_t) and pop[i,-3]>=((22*ek+1)*M_t):
                    M_t= random.randrange(20,50,1)

            M_pop = (M_MC1, M_MC2, M_MC3, M_MC4, M_MC5, M_W1, M_W2, M_W3, M_W4, M_W5, \
                     M_MB1, M_MB2, M_MB3,M_MB4,M_MB5,M_h, M_w, M_t,OT_FG)
            pop[i,int(mutate_point)]=M_pop[int(mutate_point)]

    return pop


#Main Function
start_time = time.time()
pop = np.zeros([POP_SIZE, DNA_SIZE])
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
fitness_recorded=np.zeros((N_GENERATIONS, 1))
pop_recorded=np.zeros((N_GENERATIONS, DNA_SIZE))

for _ in range(N_GENERATIONS):
    num=_
    realchangers = translateDNA(pop)
    r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc = F(num, realchangers)
    fitness = get_fitness(r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc)
    print('start')
    print('Fitness is',fitness)
    print('end')
    pop = select(pop, r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc,fitness)
    pop = crossover(pop, r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc, fitness)
    pop = mutate(pop, r_C1, r_C2,r_C3,r_C4,r_C5, r_W1, r_W2,r_W3,r_W4,r_W5, r_BM,story_max, money_consc, fitness)
    print('Most fitted NDA now:', pop[np.argmin(fitness),:])
    print('Results are：','\nr_C：',r_C1[np.argmin(fitness)],r_C2[np.argmin(fitness)],r_C3[np.argmin(fitness)],\
          r_C4[np.argmin(fitness)],r_C5[np.argmin(fitness)],'\nr_W：',r_W1[np.argmin(fitness)],r_W2[np.argmin(fitness)],\
          r_W3[np.argmin(fitness)],r_W4[np.argmin(fitness)],r_W5[np.argmin(fitness)],'\nr_BM：',r_BM[np.argmin(fitness)],\
          '\nstory_max：',story_max[np.argmin(fitness)],'\nmoney：',money_consc[np.argmin(fitness)])
    fitness_recorded[_]= float(fitness[np.argmin(fitness)])
    pop_recorded[_] = pop[np.argmin(fitness)]
    a=1

allresult = np.concatenate((fitness_recorded,pop_recorded),axis=1)
#You can fill in the storage location according to your needs 

savepath = 'allresult50-5.xlsx'
allresult = pd.DataFrame(allresult)
writer = pd.ExcelWriter(savepath)
allresult.to_excel(writer,'Sheet2')
writer.save()
plt.plot(list(range(N_GENERATIONS)), list(fitness_recorded))
plt.xlabel("Iterations")
plt.ylabel("Construction money")
plt.show()
end_time = time.time()
print('Time:',end_time-start_time)