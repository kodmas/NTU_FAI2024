# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import copy
def minCostConnectPoints(points):
    # credits: CSDN pages. used as a utility function to calculate MST of a set of points.
    """
    :type points: List[List[int]]
    :rtype: int
    """
    # standard definition of union find
    def find(p):
        while p != roots[p]:
            p = roots[p]
        return p
    def union(p,q):
        roots[find(p)] = find(q)
        
    roots = list(range(len(points)))
    
    graph = []
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            dis = abs(points[i][0]-points[j][0]) + abs(points[i][1]-points[j][1])
            graph.append((i,j,dis))
    
    # sort the graph
    graph.sort(key=lambda x:x[2])
    cost = 0
    num_edge = 0
    for p,q,dis in graph:
        if num_edge == len(points)-1:
            break
        # if p and q is already connected, we skip this edge
        if find(p)==find(q):
            continue
        # if q and q are not connected, we connect them and add to cost
        union(p,q)
        cost += dis
        num_edge += 1
    return cost

class Node:
    def __init__(self, position, forward_cost, backward_cost, objectives, parent = None):
        self.position = position
        self.forward_cost = forward_cost
        self.backward_cost = backward_cost
        self.total_cost = forward_cost + backward_cost
        self.objectives = objectives
        self.parent = parent # another node
    
    def __str__(self):
        return f"position: {self.position}\n forward_cost:{self.forward_cost}\n objectives: {self.objectives}\n"
        # print(self.parent)

class State:
    def __init__(self, node, position = None, objectives = None):
        if position is None:
            self.position = node.position
            self.objectives = node.objectives
        else:
            self.position = position
            self.objectives = objectives
    
class BinaryHeap:
    # credits: ChatGPT. used to get a Heap of Node type as a Priority Queue.
    def __init__(self):
        self.heap_list = [Node((0, 0), 0, 0, [], 0)]
        self.current_size = 0

    def insert(self, item):
        self.heap_list.append(item)
        self.current_size += 1
        self.perc_up(self.current_size)

    def perc_up(self, i):
        while i // 2 > 0:
            if self.heap_list[i].total_cost < self.heap_list[i // 2].total_cost:
                temp = self.heap_list[i // 2]
                self.heap_list[i // 2] = self.heap_list[i]
                self.heap_list[i] = temp
            i //= 2

    def del_min(self):
        ret_val = self.heap_list[1]
        self.heap_list[1] = self.heap_list[self.current_size]
        self.current_size -= 1
        self.heap_list.pop()
        self.perc_down(1)
        return ret_val

    def perc_down(self, i):
        while (i * 2) <= self.current_size:
            mc = self.min_child(i)
            if self.heap_list[i].total_cost > self.heap_list[mc].total_cost:
                temp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = temp
            i = mc

    def min_child(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_list[i * 2].total_cost < self.heap_list[i * 2 + 1].total_cost:
                return i * 2
            else:
                return i * 2 + 1

def calc_manhattan(pos1, pos2):
    return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])

def calc_corner(position, Objectives):
    # return 0
    
    # if(len(Objectives) == 0):
    #     return 0
    # result = 0
    # for i in Objectives:
    #     result += calc_manhattan(i, position)
    # return result/len(Objectives)
    
    # if(len(Objectives) == 0):
    #     return 0
    # result = 0
    # for i in Objectives:
    #     result = max(result, calc_manhattan(i, position))
    # return result

    return minCostConnectPoints([position, *Objectives])
    
def calc_full(position, Objectives):
    # return 0
    
    # if(len(Objectives) == 0):
    #     return 0
    # result = 0
    # for i in Objectives:
    #     result += calc_manhattan(i, position)
    # return result/len(Objectives)
    
    # if(len(Objectives) == 0):
    #     return 0
    # result = 0
    # for i in Objectives:
    #     result = max(result, calc_manhattan(i, position))
    # return result

    return minCostConnectPoints([position, *Objectives])
    
def calc_heuristic(problem, position, Objectives):
    if problem == "base":
        return calc_manhattan(position, Objectives[0])
    elif problem == "corner":
        return calc_corner(position, Objectives)
    elif problem == "full":
        return calc_full(position, Objectives)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    init = Node(maze.getStart(), 0, 0, maze.getObjectives())
    init.parent = init
    
    queue = []
    visited = {} # saving the visited states
    queue.append(init) # enqueue start node
    while len(queue) != 0:
        curr = queue.pop(0)
        if maze.getObjectives()[0] == curr.position: # goal test
            ans = []
            ans.insert(0, curr.position)
            while curr.position is not maze.getStart():
                curr = curr.parent
                ans.insert(0, curr.position)
            return ans
        
        if(visited.get(curr.position) != curr.objectives):
            visited[curr.position] = curr.objectives
            for _, i in enumerate(maze.getNeighbors(*curr.position)): # fringe expansion
                node = Node(i, 0, 0, maze.getObjectives(), curr)
                queue.append(node)
            
    return []

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    init = Node(
        maze.getStart(), 
        0, 
        calc_heuristic("base", maze.getStart(), maze.getObjectives()), 
        maze.getObjectives()
    )
    init.parent = init
    
    fringe = BinaryHeap()
    visited = {} # saving the visited states
    fringe.insert(init) # enqueue start node
    
    while fringe.current_size != 0:
        curr = fringe.del_min()
        
        if maze.getObjectives()[0] == curr.position: # goal test
            ans = []
            ans.insert(0, curr.position)
            while curr.position is not maze.getStart():
                curr = curr.parent
                ans.insert(0, curr.position)
            return ans
        
        if(visited.get(curr.position) != curr.objectives):
            visited[curr.position] = curr.objectives
            for _, i in enumerate(maze.getNeighbors(*curr.position)): # fringe expansion
                node = Node(
                    i, 
                    curr.forward_cost+1, 
                    calc_heuristic("base", i, maze.getObjectives()),  
                    curr.objectives, 
                    curr
                )
                fringe.insert(node)
            
    return []

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    init = Node(
        maze.getStart(), 
        0, 
        calc_heuristic("corner", maze.getStart(), maze.getObjectives()), 
        maze.getObjectives()
    )
    init.parent = init
    
    fringe = BinaryHeap()
    visited = set() # saving the visited states
    fringe.insert(init) # enqueue start node
    
    while fringe.current_size != 0:
        curr = fringe.del_min()
        
        if len(curr.objectives) == 0: # goal test
            ans = []
            ans.insert(0, curr.position)
            while curr.parent is not curr:
                curr = curr.parent
                ans.insert(0, curr.position)
            return ans
        
        if frozenset([f"curr: {curr.position}", *curr.objectives]) not in visited:   
            visited.add(frozenset([f"curr: {curr.position}", *curr.objectives]))
            for _, i in enumerate(maze.getNeighbors(*curr.position)): # fringe expansion
                if i in curr.objectives:
                    objective_list = copy.deepcopy(curr.objectives)
                    objective_list.remove(i)
                else:
                    objective_list = copy.deepcopy(curr.objectives)
                node = Node(
                    i, 
                    curr.forward_cost+1, 
                    calc_heuristic("corner", i, objective_list),  
                    objective_list, 
                    curr
                )
                fringe.insert(node)
            
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    init = Node(
        maze.getStart(), 
        0, 
        calc_heuristic("full", maze.getStart(), maze.getObjectives()), 
        maze.getObjectives()
    )
    init.parent = init
    
    fringe = BinaryHeap()
    visited = set() # saving the visited states
    fringe.insert(init) # enqueue start node
    
    while fringe.current_size != 0:
        curr = fringe.del_min()
        
        if len(curr.objectives) == 0: # goal test
            ans = []
            ans.insert(0, curr.position)
            while curr.parent is not curr:
                curr = curr.parent
                ans.insert(0, curr.position)
            return ans
        
        if frozenset([f"curr: {curr.position}", *curr.objectives]) not in visited:   
            visited.add(frozenset([f"curr: {curr.position}", *curr.objectives]))
            for _, i in enumerate(maze.getNeighbors(*curr.position)): # fringe expansion
                if i in curr.objectives:
                    objective_list = copy.deepcopy(curr.objectives)
                    objective_list.remove(i)
                else:
                    objective_list = copy.deepcopy(curr.objectives)
                node = Node(
                    i, 
                    curr.forward_cost+1, 
                    calc_heuristic("full", i, objective_list),  
                    objective_list, 
                    curr
                )
                fringe.insert(node)
    return []

def closest_point(start, candidates):
    least_dist = 99999999
    for i in candidates:
        if(calc_manhattan(start, i)<least_dist):
            least_dist = calc_manhattan(start, i)
            candidate = i
    return candidate
    
def astar_path(maze, start, objectives):
    # first calculate target
    target = closest_point(start, objectives)
    
    init = Node(
        start, 
        0, 
        calc_manhattan(start, target), 
        [target]
    )
    init.parent = init
    
    fringe = BinaryHeap()
    visited = {} # saving the visited states
    fringe.insert(init) # enqueue start node
    
    while fringe.current_size != 0:
        curr = fringe.del_min()
        
        if target == curr.position: # goal test
            ans = []
            ans.insert(0, curr.position)
            while curr.position is not start:
                curr = curr.parent
                ans.insert(0, curr.position)
            for point in ans:
                if point in objectives:
                    objectives.remove(point)
            return (ans, target, objectives)
        
        if(visited.get(curr.position) != target):
            visited[curr.position] = target
            for _, i in enumerate(maze.getNeighbors(*curr.position)): # fringe expansion
                node = Node(
                    i, 
                    curr.forward_cost+1, 
                    calc_manhattan(i, target),  
                    [target], 
                    curr
                )
                fringe.insert(node)

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # find next state which targets the nearest manhattan distance. eat the food passed by.
    objectives = maze.getObjectives()
    start = maze.getStart()
    
    path = []
    while len(objectives) != 0:
        new_path, new_pos, new_objectives = astar_path(maze, start, objectives)
        for idx, i in enumerate(new_path):
            if(start == maze.getStart()):
                path.append(i)
            else:
                if(idx != 0):
                    path.append(i)
        start = new_pos
        objectives = new_objectives
    return path