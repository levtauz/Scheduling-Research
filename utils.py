"""
Utilities that are commenly used
"""
import matplotlib.pyplot as plt
import numpy as np
import time
import operator 

class Timer:
    """
    Used with a with statement to track time performance
    """
    def __enter__(self):
                    self.start = time.clock()
                    return self

    def __exit__(self, *args):
                    self.end = time.clock()
                    self.elapsed = self.end - self.start

def plot_LBP_out(x):
    """
    Plots the output of the LPB arrays of the x and y values
    Expects input to be arrange in cases,room,day,time
    Restrict to small number of dimensions
    """
    x = np.array(x)
    assert x.ndim == 4,"input should be 4D"
    cols = 3
    rows = np.prod(x.shape[1:-1]) *1.0/cols + 1
    for case in x:
        for room_i in range(len(case)):
            for day_i in range(len(case[room_i])):
                plt.subplot(cols,rows,room_i+ day_i*len(case[room_i]))
                plt.stem(case[room_i,day_i])
    plt.show()

def schedule_cost(schedule, samples, weights,overtime_limit = 10,rooms = 1 , days = 1):
	"""
	Computes the average cost of a schedule
	Inputs:
			Schedule- List where index i represents the scheduled time of case i
			samples - List of duration samples
			weights - weight of [idle time, delay, overtime]
	Output:
			Cost - float
	"""
	#Extract weights 
	w_i = weights[0] #idle weight
	w_d = weights[1] #delay weight
	w_o = weights[2] #overtime weight

	#initialize array that stores cases by room and day
	#initilize empty array
	D = [[[] for _ in range(days)] for _ in range(rooms)] 
	
	#assign to rooms and days
	for i in range(len(schedule)):
		if schedule[i] != [] :
			r = schedule[i][0] #room
			d = schedule[i][1] #day
			t = schedule[i][2] #time
			D[r][d].append((i,t)) #(case, time)
	
	#sort list
	D = [[sorted(D[i][j],key =operator.itemgetter(1)) for j in range(days)] for i in range(rooms)]
	
	#Make D a numpy array
	D= np.array(D)

	#Calcualte cost
	cost = 0
	for sample in samples:
		for r in range(rooms):
			for d in range(days):
				sched = D[r][d]
				last_end_t = 0
				for case,time in sched:
					dur = sample[case]
					cost += max(time - last_end_t  , 0) * w_i #Idle time
					cost += max(last_end_t - time, 0) * w_d #delay time
					last_end_t = max(last_end_t,time) + dur #updates end time
				cost += max(last_end_t - overtime_limit,0) * w_o #calculates overtime
	
	return cost * 1.0/len(samples)


