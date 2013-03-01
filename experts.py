import numpy as np

class MultWeights:
    
    def __init__(self,n,epsilon):
        self.n = n
        self.epsilon = epsilon
        self.weights = np.ones(n)

    def update(self,losses):
        assert(self.weights.size == losses.size)
        expected_loss = 0.0
        for i in xrange(0,self.weights.size):
            expected_loss += losses[i]*self.weights[i]
            self.weights[i] *= (1.0 - self.epsilon)**losses[i]

        expected_loss /= self.weights.sum()

        return expected_loss
    
    def expected_loss(self,losses):
        expected_loss = 0.0
        for i in xrange(0,self.weights.size):
            expected_loss += losses[i]*self.weights[i]

        expected_loss /= self.weights.sum()

        return expected_loss
    
    def get_strategy(self):
        strategy = self.weights/self.weights.sum()
        return strategy

def scale_matrix(G):
    result = G.copy()
    min_val = G[0][0]
    # min_elt = (0,0):
    max_val = G[0][0]
    for row in G:
        for elt in row:
            if (elt < min_val):
                min_val = elt
            if (elt > max_val):
                max_val = elt
            
    result -= min_val
    result /= (max_val - min_val) # hope it's not zero! 
    print "translated and scaled matrix"
    print result
    return result

def zip_multiply(strategy, payoff):
    total = 0
    for i in range(len(strategy)):
        total += strategy[i]*payoff[i]
    return total

def delta_equilibrium(row_strategy, col_strategy, game, expected_loss):
    biggest_delta = -1
    for i in range(len(row_strategy)):
        pure = zip_multiply(col_strategy, game[i])
        #print "\t\tpure strategy", i, "row", pure
	if pure < expected_loss and -(pure-expected_loss) > biggest_delta:
	    biggest_delta = -(pure-expected_loss)
    for i in range(len(col_strategy)):
        pure = zip_multiply(row_strategy, [game[0][i], game[1][i], game[2][i], game[3][i]])
	#print "\t\tpure strategy", i, "column", pure
	if pure > expected_loss and pure-expected_loss > biggest_delta:
	    biggest_delta = pure-expected_loss
    print "\t\tbiggest delta today", biggest_delta

def find_best_col(Cols, mw_framework):
    best_col = 0
    max_loss = mw_framework.expected_loss(Cols[0])
    for j in xrange(1,Cols.shape[0]):
        this_loss = mw_framework.expected_loss(Cols[j])
        if (max_loss < this_loss):
            best_col = j
            max_loss = this_loss
    return (best_col,max_loss)

def mult_weights_two_person_game(G, T, epsilon):
    mw_framework = MultWeights(G.shape[0],epsilon)
    Cols = G.transpose()
    least_loss =  1
    least_loss_strategy = mw_framework.get_strategy()
    col_strategies = []

    total_loss = 0.0

    for i in xrange(0,T):
        (best_col,max_loss) = find_best_col(Cols,mw_framework)
        total_loss += max_loss

        col_strategies.append(best_col)

        # print i, total_loss, max_loss

        if (least_loss > max_loss):
            least_loss = max_loss
            least_loss_strategy = mw_framework.get_strategy()

        expected_loss = mw_framework.update(Cols[best_col])

	if i in [10, 100, 139, 10000]:
            avg_column_strategy = np.zeros(Cols.shape[0])
            for j in xrange(0,len(col_strategies)):
                avg_column_strategy[col_strategies[j]] += 1.0/(i+1)
	    avg_loss = total_loss/(i+1)
    
	    print "***********\nday ", i
	    print "\trow strategy ", least_loss_strategy
	    print "\tcolumn strategy ", avg_column_strategy
	    print "\texpected loss ", expected_loss
	    delta_equilibrium(least_loss_strategy, avg_column_strategy, G, expected_loss)
	    print "\tleast loss", least_loss, " and max loss", max_loss
	    print "\taverage loss ", avg_loss
    
    #avg_column_strategy = np.zeros(Cols.shape[0])
    #for i in xrange(0,len(col_strategies)):
        #print "cols strategy", col_strategies[i]
        #avg_column_strategy[col_strategies[i]] += 1.0/T
    
    #print "Column Strategy", avg_column_strategy
    #print "Row Strategy", least_loss_strategy

    #avg_loss = total_loss/T
    #print "average_loss %f" % avg_loss

    
rock_paper_scissors = np.array([[0.0,1,-1],[-1,0,1],[1,-1,0],[0,-1,-1]])

mult_weights_two_person_game(scale_matrix(rock_paper_scissors),140,.1)
