import numpy as np

class Sampler(object):

    def __init__(self, types, samplers):
        assert len(types) == len(samplers), "length should be equivalent"
        self.samplers = {}
        for i in range(len(types)):
            self.samplers[types[i]] = samplers[i]
        
    def add_type(self,C,sampler):
        self.samplers[C] = sampler

    def sample(self, C, size=1):
        """
        C - type of sample
        Size - How many samples
        """
        if C not in self.samplers:
            raise ValeuError("Type {0} does not exist".format(C))
        if(size == 1):
            return np.asscalar(self.samplers[C](size))
        else:
            return self.samplers[C](size)
  

def gradient(cases,sampler,step, alpha, beta,gamma,D,T,S = 10000):
    """
    Inputs : 
    cases - cases to be scheduled
    sampler - A sampler class to abstract sampling
    step - stepsize
    alpha - change for graident descent
    beta - idle cost
    gamma - overtime cost
    D - Maximum point until overtime
    T - Warning Slack for scheduling
    S - Number of iterations
    
    Outputs:
    WT - Warning Times for each case
    K - Mapping of which surgery to warn during, -1 means no surgery
    TC - Cost over Time
    """
    N = len(cases) #Number of cases
    TC = np.zeros(S) #Total Cost
    IC = np.zeros(S) # Idle Cost
    WC = np.zeros(S) # Waiting Cost
    OC = np.zeros(S) # Overtime Cost

    WT = np.zeros((S,N)) #Warning times
    K = np.zeros((S,N)) #Mapping of which surgery to warn in
    G = np.zeros(N) #Gradient

    SE = np.zeros(N) #Expected Start times of surgeries

    # Calculate Imperical Mean and STD
    means = np.zeros(N)
    stds = np.zeros(N)
    for i in range(N):
        samp = sampler.sample(cases[i],100000)
        means[i] = np.mean(samp)
        stds[i] = np.std(samp)

    # First, Initialize the Start times using mean and std
    for i in range(1,N):
        SE[i] = SE[i-1] + means[i-1] + 0.5 * stds[i-1]

    # Second, compute warning times
    for i in range(1,N):
        WT[0,i] = max(0,SE[i] - T)
        
    #Third, find the surgeries during which the warnings should take place
    K[:,0] = -1
    for i in range(1,N):
        if WT[0,i] == 0:
            K[0,i] == -1
        else:
            for k in range(i):
                if(SE[k] < WT[0,i]):
                    K[0,i] = k
            WT[0,i] =  WT[0,i] - SE[K[0,i]]

    SD = np.zeros(N) #Surgery Durations
    ST = np.zeros((S,N)) #Start times of surgeries
    Y = np.zeros((S,N)) #End time of Surgaries
    A = np.zeros(N)
    B = np.zeros(N)
    C = np.zeros(N)

    for s in range(S-1):
        
        #First, Generate Samples
        for i in range(N):
            SD[i] = sampler.sample(cases[i])
            
        #Second, generate start and end times based of Warning Times and Durations
        Y[s,0] = SD[0]
        for i in range(1,N):
            if K[s,i] == -1:
                ST[s,i] = Y[s,i-1]
            else:
                scheduled = ST[s,K[s,i]] + min(WT[s,i],SD[K[s,i]]) + T
                ST[s,i] = max(Y[s,i-1], scheduled)
            Y[s,i] =  ST[s,i] + SD[i]
         
        #Third,Calculate Average Cost
        for i in range(1,N):
            if K[s,i] != -1:
                temp = ST[s,K[s,i]]+min(WT[s,i],SD[K[s,i]]) + T - Y[s,i-1]
                V = max(0,temp)
            else:
                V = max(0,WT[s,i]+T-Y[s,i-1])
            WC[s+1] = (s*WC[s]+V)/(s+1)
            IC[s+1] = (s*IC[s]+beta*max(0,ST[s,i]-Y[s,i-1]))/(s+1)
        OC[s+1] = (s*IC[s]+gamma*max(0,Y[s,N-1] - D))/(s+1)
        TC[s+1] = WC[s+1] + IC[s+1] + OC[s+1] 
        
   
        #Fourth, Compute A,B, and C
        BC = np.ones(N)
        for i in range(1,N):
            if(ST[s,i] == Y[s,i-1]):
                A[i] = 1
            else:
                A[i] = 0
                
        for i in range(N-1,0,-1):
            BC[i] = BC[i]*(1-A[i])
            B[i] = 1-BC[i]
        
        for i in range(N):
            if(Y[s,N-1] >  D):
                C[i] = BC[i]
            else:
                C[i] = 0;
                
        #Update Gradient
        for i in range(1,N):
            G[i] = -alpha*A[i] + beta*(1-A[i])*BC[i] + gamma*(1-A[i])*C[i]
        
        #Update Warning Times
        for i in range(1,N):
            WT[s+1,i] = max(0,WT[s,i] - step * G[i])
            
        #Check for Negative warning times
        for i in range(1,N):
            if WT[s+1,i] == 0:
                if K[s,i] == -1:
                    K[s+1,i] = -1
                else:
                    K[s+1,i] = K[s,i]-1
                    if(K[s+1,i] != -1):
                        WT[s+1,i] = means[K[s+1,i]]
                    else:
                        WT[s+1,i] = 0
            else:
                K[s+1,i] = K[s,i]
                
        #Check for increasing WT
        for i in range(1,N):
            if WT[s+1,i] > SD[i]:
                WT[s+1,i] = 1
                K[s+1,i] = min(K[s,i] + 1 ,i-1)
    
    return WT, K, TC,WC,IC,OC

def gradient_2(cases,sampler,step, alpha, beta,gamma,D,T,S = 10000):
    """
    Inputs : 
    cases - cases to be scheduled
    sampler - A sampler class to abstract sampling
    step - stepsize
    alpha - change for graident descent
    beta - idle cost
    gamma - overtime cost
    D - Maximum point until overtime
    T - Warning Slack for scheduling
    S - Number of iterations
    
    Outputs:
    WT - Warning Times for each case
    K - Mapping of which surgery to warn during, -1 means no surgery
    TC - Cost over Time
    """
    N = len(cases) #Number of cases
    TC = np.zeros(S) #Total Cost
    IC = np.zeros(S) # Idle Cost
    WC = np.zeros(S) # Waiting Cost
    OC = np.zeros(S) # Overtime Cost

    WT = np.zeros((S,N)) #Warning times
    K = np.zeros((S,N)) #Mapping of which surgery to warn in
    G = np.zeros(N) #Gradient

    SE = np.zeros(N) #Expected Start times of surgeries

    # Calculate Imperical Mean and STD
    means = np.zeros(N)
    stds = np.zeros(N)
    for i in range(N):
        samp = sampler.sample(cases[i],100000)
        means[i] = np.mean(samp)
        stds[i] = np.std(samp)

    # First, Initialize the Start times using mean and std
    for i in range(1,N):
        SE[i] = SE[i-1] + means[i-1] + 0.5 * stds[i-1]

    # Second, compute warning times
    for i in range(1,N):
        WT[0,i] = max(0,SE[i] - T)
        
    #Third, find the surgeries during which the warnings should take place
    K[:,0] = -1
    for i in range(1,N):
        if WT[0,i] == 0:
            K[0,i] == -1
        else:
            for k in range(i):
                if(SE[k] < WT[0,i]):
                    K[0,i] = k
            WT[0,i] =  WT[0,i] - SE[K[0,i]]

    SD = np.zeros(N) #Surgery Durations
    ST = np.zeros((S,N)) #Start times of surgeries
    Y = np.zeros((S,N)) #End time of Surgaries
    
    for s in range(S-1):
        H = np.zeros(N) # Head of chain indicator
        P = np.zeros(N) # Count of surgeries in chain after a case that depend on a case before
        O = np.zeros(N) # If current chain ends after D

        #First, Generate Samples
        for i in range(N):
            SD[i] = sampler.sample(cases[i])
            
        #Second, generate start and end times based of Warning Times and Durations
        Y[s,0] = SD[0]
        for i in range(1,N):
            if K[s,i] == -1:
                ST[s,i] = max(Y[s,i-1], np.sum(means[:i]))
            else:
                scheduled = ST[s,K[s,i]] + min(WT[s,i],SD[K[s,i]]) + T
                ST[s,i] = max(Y[s,i-1], scheduled)
            Y[s,i] =  ST[s,i] + SD[i]
         
        #Third,Calculate Average Cost
        for i in range(1,N):
            if K[s,i] != -1:
                temp = ST[s,K[s,i]]+min(WT[s,i],SD[K[s,i]]) + T - Y[s,i-1]
                V = max(0,temp)
            else:
                V = max(0,WT[s,i]+T-Y[s,i-1])
            WC[s+1] = (s*WC[s]+V)/(s+1)
            IC[s+1] = (s*IC[s]+beta*max(0,ST[s,i]-Y[s,i-1]))/(s+1)
        OC[s+1] = (s*IC[s]+gamma*max(0,Y[s,N-1] - D))/(s+1)
        TC[s+1] = WC[s+1] + IC[s+1] + OC[s+1] 
        
   
        #Fourth, Compute H,P, and O
        #Find chains
        chains = []
        cur = [0]
        for i in range(1,N):
            if(ST[s,i] == Y[s,i-1]):
                cur.append(i)
            else:
                chains.append(cur)
                cur = [i]
        chains.append(cur)

        for chain in chains:
            H[chain[0]] = 1
            for i in range(len(chain)):
                P[chain[i]] = np.sum(K[s,chain[i:]] < chain[i])
                O[chain[i]] = np.int0(O[chain[-1]] > D)
        
        #Update Gradient
        for i in range(1,N):
            if K[s,i] == -1:
                G[i] = 0 
            else:
                G[i] = alpha*P[i] + -alpha*(1-H[i])+ beta*H[i]*O[i] + gamma * O[i]
        
        #Update Warning Times
        for i in range(1,N):
            WT[s+1,i] = max(0,WT[s,i] - step * G[i])
            
        #Check for Negative warning times
        for i in range(1,N):
            if WT[s+1,i] == 0:
                if K[s,i] == -1:
                    if ST[s,i] - T <= 0:
                        K[s+1,i] == -1
                    else:
                        for k in range(i):
                            if(ST[s,k] < ST[s,i] - T):
                                K[s+1,i] = k
                        WT[s+1,i] =  ST[s,i] - T - ST[K[s+1,i]]
                else:
                    K[s+1,i] = K[s,i]-1
                    if(K[s+1,i] != -1):
                        WT[s+1,i] = means[K[s+1,i]]
                    else:
                        WT[s+1,i] = 0
            elif WT[s+1,i] > SD[i]:
                K[s+1,i] = min(K[s,i] + 1 ,i-1)
                if(i-1 > K[s,i] + 1):
                    W[s+1,i] = step
            else:
                K[s+1,i] = K[s,i]
                
    return WT, K, TC,WC,IC,OC

def calculate_cost(WT,K,cases,sampler,beta, gamma,T,D,M=1000):
    """

    """
    
    #Sample Test Set
    N = len(cases)
    test_set = np.zeros((M,N))
    for i in range(N):
        test_set[:,i] = sampler.sample(cases[i],M)
    
    # Calculate Imperical Mean and STD
    means = np.zeros(N)
    stds = np.zeros(N)
    for i in range(N):
        samp = sampler.sample(cases[i],100000)
        means[i] = np.mean(samp)
        stds[i] = np.std(samp)


    Start = np.zeros((M,N))
    End = np.zeros((M,N))

    End[:,0] = test_set[:,0]
    for i in range(1,N):
        if K[i] == -1:
            Start[:,i] = np.maximum(End[:,i-1], np.sum(means[:i]))
        else:
            scheduled = Start[:,K[i]] + np.minimum(WT[i],test_set[:,K[i]]) + T
            Start[:,i] = np.maximum(End[:,i-1], scheduled)
        End[:,i] =  Start[:,i] + test_set[:,i]

    #Calculate Average Cost
    Waiting = 0
    Idle = 0
    for i in range(1,N):
        if K[i] != -1:
            temp = Start[:,K[i]]+np.minimum(WT[i],test_set[:,K[i]]) + T - End[:,i-1]
            V = np.maximum(0,temp)
        else:
            V = np.maximum(0,WT[i]+T-End[:,i-1])
        Waiting += np.sum(V)/M
        Idle += np.sum(beta*np.maximum(0,Start[:,i]-End[:,i-1]))/M
        
    Over = np.sum(gamma*np.maximum(0,End[:,-1] - D))/M
    Total = Waiting +Idle+Over

    return Total,Waiting,Idle,Over
