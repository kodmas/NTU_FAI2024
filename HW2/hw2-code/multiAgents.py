# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        # positive toward food
        if len(newFood.asList()) > 0:
            fooddist = [manhattanDistance(food, newPos) for food in newFood.asList()]
            score += 1/min(fooddist)
        

        # negative toward ghost and try to eat ghost while powered
        if len(newGhostStates) > 0:
            ghostdist = [manhattanDistance(ghost.configuration.pos, newPos) for ghost in newGhostStates]
            if sum(newScaredTimes)!= 0 and min(ghostdist) <= min(newScaredTimes) and min(newScaredTimes) != 0:
                score += 1/min(ghostdist)
            else:
                if min(ghostdist) != 0:
                    score -= 1/min(ghostdist)
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        state = gameState
        ghostIdx = [i for i in range(1,state.getNumAgents())]
        
        def max_value(state,depth):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            value = -math.inf
            for action in state.getLegalActions(0):
                value = max(value, min_value(state.generateSuccessor(0,action),depth,ghostIdx[0]))
            return value
        def min_value(state,depth,ghost):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            value = math.inf
            for action in state.getLegalActions(ghost):
                if ghost == ghostIdx[-1]:
                    value = min(value,max_value(state.generateSuccessor(ghost,action),depth+1))
                else:
                    value = min(value,min_value(state.generateSuccessor(ghost,action),depth,ghost+1))
            return value
        
        res = [(action, min_value(gameState.generateSuccessor(0, action), 0, ghostIdx[0])) for action in gameState.getLegalActions(0)]
        res.sort(key=lambda k: k[1])

        return res[-1][0]
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        state = gameState
        ghostIdx = [i for i in range(1,state.getNumAgents())]
        alpha,beta = -math.inf,math.inf
        def node_value(state, depth, agentidx,alpha,beta):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            if agentidx == 0:
                return max_value(state, depth,alpha,beta)
            else:
                return min_value(state, depth, agentidx,alpha,beta) 
        def max_value(state,depth,alpha,beta):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            value = -math.inf
            for action in state.getLegalActions(0):
                value = max(value, node_value(state.generateSuccessor(0,action),depth,1,alpha,beta))
                if value > beta:
                    return value
                alpha = max(alpha,value)
            return value
        def min_value(state,depth,ghost,alpha,beta):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            value = math.inf
            for action in state.getLegalActions(ghost):
                if ghost == ghostIdx[-1]:
                    value = min(value,node_value(state.generateSuccessor(ghost,action),depth+1,0,alpha,beta))
                else:
                    value = min(value,node_value(state.generateSuccessor(ghost,action),depth,ghost+1,alpha,beta))
                if value < alpha:
                    return value
                beta = min(beta,value)
            return value
        maxvalue = -1e9
        max_action = Directions.STOP

        for action in gameState.getLegalActions(0):
            suc_value = node_value(gameState.generateSuccessor(0, action), 0, 1,alpha,beta)
            if suc_value > maxvalue:
                maxvalue, max_action = suc_value, action

            alpha = max(alpha, maxvalue)
        return max_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def node_value(state, depth, agentidx):
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            if agentidx == 0:
                return max_value(state, depth)
            else:
                return exp_value(state, depth, agentidx) 
        
        def max_value(state,depth):
            value = -math.inf
            for action in state.getLegalActions(0):
                value = max(value, node_value(state.generateSuccessor(0,action),depth,1))
            return value
        def exp_value(state,depth,ghost):
            ghostNums = state.getNumAgents() - 1
            if(state.isWin() or state.isLose() or depth == self.depth):
                return self.evaluationFunction(state)
            value = 0
            for action in state.getLegalActions(ghost):
                if ghost == ghostNums:
                    value += node_value(state.generateSuccessor(ghost,action),depth + 1,0)
                else:
                    value += node_value(state.generateSuccessor(ghost,action),depth,ghost+1)
            value /= len(state.getLegalActions(ghost))
            return value
        
        maxvalue = -1e9
        max_action = Directions.STOP

        for action in gameState.getLegalActions(0):
            suc_value = node_value(gameState.generateSuccessor(0, action), 0, 1)
            if suc_value > maxvalue:
                maxvalue, max_action = suc_value, action

        return max_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    currFood = currentGameState.getFood()
    currPos = currentGameState.getPacmanPosition()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    # positive toward food
    if len(currFood.asList()) > 0:
        fooddist = [manhattanDistance(food, currPos) for food in currFood.asList()]
        score += 1/min(fooddist)
    else:
        score += 1
    

    # negative toward ghost and try to eat gost while powered
    if len(currGhostStates) > 0:
        ghostdist = [manhattanDistance(ghost.configuration.pos, currPos) for ghost in currGhostStates]
        if min(ghostdist) == 0:
            return -math.inf
        if sum(currScaredTimes)!= 0 and min(currScaredTimes) != 0:
            score += 60/min(ghostdist)
        else:
            score -= 1/min(ghostdist)
    
    return score

# Abbreviation
better = betterEvaluationFunction
