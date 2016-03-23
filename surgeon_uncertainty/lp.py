from pulp import * 
import numpy as np
import matplotlib.pyplot as plt
import sys
from ..utils import Timer

def sample_scheduler(samples,weights, over_weight = 1, max_time = 60*10,debug = False):
    """
    Inputs:
        Samples - 2D array of samples: row= sample, column= duration of case in index i
        Weights - weights for each surgeries waiting time and idle time
        Over_Weight - scalar for the overall overtime 
        max_time - time when overtime is counted
    Output:
        Schedule - returns a list of times each surgery in index i should start
        Cost - Cost of schedule
    
    Refer to SurgerySequencing.pdf
    """
    (K,N) = samples.shape #(Number of samples, Number of surgeries)
    assert weights.shape == (2,N),"need enough weights for waiting and idling for all variables"

    w_weights = weights[0]
    s_weights = weights[1]
    
    prob = LpProblem("schedule",LpMinimize)
    
    ###Variable Creation###
    w = [] #waiting time 
    s = [] #idling time
    l = [] #over time (lateness)
    g = [] #earliness (slack variable)
    x = np.empty( N,'object') ##amount of time scheduled for
    for i in range(N):
        x[i] = LpVariable("x {0}".format(i), lowBound = 0,cat='Continuous')
    
    for i in range(K):
        w.append([LpVariable("w {0} {1}".format(i,j), lowBound = 0,cat='Continuous') for j in range(N)])
        s.append([LpVariable("s {0} {1}".format(i,j), lowBound = 0,cat='Continuous') for j in range(N)])
        l.append(LpVariable("l {0}".format(i), lowBound = 0, cat = 'Continous'))
        g.append(LpVariable("g {0}".format(i), lowBound = 0, cat = 'Continous'))


    ###Objective Formulation###
    obj = []
    for i in range(K):
        obj.append(lpSum([w_weights[j]*w[i][j] for j in range(N)]))
        obj.append(lpSum([s_weights[j]*s[i][j] for j in range(N)]))
    obj.append(lpSum(l) * over_weight)
    prob += lpSum(obj) * 1.0/K
    
    ###Constraint Formulation###
    for i in range(K):
        #Relate waiting and idling time 
        prob += w[i][0] == 0
        prob += s[i][0] == 0
        for j in range(1,N):
            prob += w[i][j] - s[i][j] == w[i][j-1] + samples[i][j-1] - x[j-1]  
        
        #Lateness constraint
        prob += -w[i][-1] + l[i] - g[i] == samples[i][-1] - max_time + np.sum(x[:-1])
    
    ###Solving###
    #prob.solve(GLPK(msg =0))
    prob.solve() 
    
    if(debug): 
        print "Schedule Amount: ",[value(p) for p in x]
        print "Status: " , LpStatus[prob.status]
        print "Obj: " , value(prob.objective)

    ###Extracting Scheduling time###
    schedule = []
    schedule.append(0)
    for i in range(1,N):
        schedule.append(sum([value(p) for p in x[0:i]]))

    return schedule, value(prob.objective)


def LBP_samples(samples,rooms, days, T, weights=[1,1],alpha = 0,debug = False):
	"""
	LBP Wrapper that computes sample mean and std and then calls LBP
	"""
	means = np.mean(samples,axis = 0)
	std = np.std(samples,axis = 0)
	return LBP(np.ceil(means + std*alpha), rooms,days, T, weights,debug)

def LBP(durations, rooms, days, T, weights = [1,1] ,debug = False):
    """
    Linear Integer/Binary Program Solver
    Given the duration of cases, the # of rooms, and the # of days,
    determine a schedule for surgeries.
    Time is discrete and if a surgery is run after time T, then it incurs an overtime penalty.
    
    Only considering the overtime and idling time
    weights = [idle,overtime]
	"""
    durations = np.array(durations)
    assert len(weights) == 2, "need two weights for overtime and idle time"
    assert durations.ndim == 1,"durations should be a 1D vector of durations"
    assert rooms > 0,"number of rooms needs to be positive"
    assert days > 0,"number of days needs to be positive"
    assert T > 0, "number of timesteps neeeds to be positive"
    
    with Timer() as tim:
        cases = len(durations) 
        maxT = T + T/4 
        prob = LpProblem("LBP",LpMinimize)
       
        ### Variable Creation ###
        with Timer() as tim_create:
            x = np.empty((cases,rooms,days,maxT),'object') # start times, indexed by [case #, room, day, time]
            y = np.empty((cases,rooms,days,maxT),'object') # indicator that case i in room j on day d at time t
            o = np.empty(cases,'object') # overtime 
            for i in range(cases):
                for j in range(rooms):
                    for d in range(days):
                        for t in range(maxT):
                            x[i,j,d,t] = LpVariable("x {0} {1} {2} {3}".format(i,j,d,t),cat='Binary')
                            y[i,j,d,t] = LpVariable("y {0} {1} {2} {3}".format(i,j,d,t),cat='Binary')
                o[i] = LpVariable("o {0}".format(i),cat='Integer')
        if(debug):
            print "=> elapsed time for creation of variables = %s s" % tim_create.elapsed


        ###Objective Formulation###
        with Timer() as tim_obj:
            obj = []
            obj.append(weights[1] * np.sum(o)) #overtime
            for i in range(cases):
                for j in range(rooms):
                    obj.append(-weights[0] * np.sum(y[i,j,:,:T])) #idle Time
            prob += lpSum(obj) + days*rooms*T 
        if (debug):
            print "=> elapsed time for obj formulation = %s s" % tim_obj.elapsed

        ###Constraint Formulation###
        with Timer() as tim_con:
            for i,j,d,t in itertools.product(range(cases),range(rooms),range(days),range(maxT)):
                a=np.sum(x[i,j,d,:t+1]) 
                b=np.sum(x[i,j,d,0:max(0,t+1-durations[i])])
                prob += y[i,j,d,t] == a-b
            for i in range(cases):
                prob += o[i] == np.sum(y[i,:,:,T:])
                temp = []
                for j in range(rooms):
                    temp.append(np.sum(x[i,j]))
                prob += lpSum(temp) <=1

            for j,d,t in itertools.product(range(rooms),range(days),range(maxT)):
                prob += np.sum(y[:,j,d,t]) <= 1
        if(debug):
            print "=> elapsed time for constraint creation = %s s" % tim_con.elapsed

    if(debug):    
        print "=> elapsed time until calling solve = %s s" % tim.elapsed
    
    ###Solving###
    #prob.solve(solver=PULP_CBC_CMD)
    with Timer() as tim:
        prob.solve()
    
    if(debug):    
        print "=> elapsed time for solving lp = %s s" % tim.elapsed
    
    ###Extracting Data###
    with Timer() as tim:
        schedule =  np.zeros(cases,dtype = 'object')
        #Use if you need to have the x and y values for debugging, currently it only constructs them
        
        x_values = np.zeros(x.shape)
        y_values = np.zeros(y.shape) #for debug
        
        for n,m,j,k in np.nditer([x,y,x_values,y_values],flags = ["refs_ok"],op_flags = ["readwrite"]):
            j[...] = n[()].value()
            k[...] = m[()].value()
        
        #Extracts the scheduled times for each case in the order that was given 
        #Format is [scheduled room, scheduled day, scheduled time]
        #If not scheduled, case is a None type
        for i in range(cases):
            indices = np.transpose(x_values[i].nonzero()) 
            if len(indices) > 0 :
                schedule[i] = indices[0]
            else:
                schedule[i] = []

    if(debug):    
        print "=> elapsed time for post-procesing data = %s s" % tim.elapsed
         
    if (debug):  
        print "objective =",value(prob.objective)

    return schedule,value(prob.objective)



if __name__ == "__main__":
    num_cases = 10
    num_samples = 10
    ranges = [(0,5),(0,5),(0,5),(0,5),(0,5), \
        (0,5),(0,5),(0,5),(0,5),(0,5)]
    samples = np.empty((num_samples,num_cases), 'object') 
    for k in range(num_samples):
        for i in range(num_cases):
            samples[k][i] = np.random.random_integers(ranges[i][0],ranges[i][1]) 
    weights = np.empty((2,num_cases),'object')
    for x in range(2):
        for i in range(num_cases):
            if x == 0:
                weights[x][i] = 0.5
            else:
                weights[x][i] = 0.5
    schedule_vals = sample_scheduler(samples, weights) 
    schedule = schedule_vals[0]
    i = 1
    for item in schedule:
        print "scheduled time for surgery " + str(i) + \
                ": " + str(item) + "\n"
        i += 1

    #Can be used for debugging
    if len(sys.argv) == 1:
        D = np.array([2]*5)
        print(D)
        prob = LBP(D,1,1,10)
        
        i = 1
        for item in prob[0]:
            print("Case " + str(i))
            if item is not None:
                print("Room: " + str(item[0]))
                print("Day: " + str(item[1]))
                print("Time: " + str(item[2]))
            else:
                print("No valid schedule")
            print("\n")
            i += 1
    """
    elif sys.argv[1] == "sanity": #Used as a sanity check for the lp and serve as a quick example 
        prob = LpProblem("test1", LpMinimize)

        # Variables
        x = LpVariable("x", 0, 4)
        y = LpVariable("y", -1, 1)
        z = LpVariable("z", 0)

        # Objective
        prob += x + 4*y + 9*z

        # Constraints
        prob += x+y <= 5
        prob += x+z >= 10
        prob += -y+z == 7

        prob.solve()

        # Solution
        for v in prob.variables():
            print v.name, "=", v.varValue
            print "objective=", value(prob.objective)
    """
