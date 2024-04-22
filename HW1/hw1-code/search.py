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

from collections import deque
from maze import Maze
import heapq
import copy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze:Maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    path = []
    (startX,startY) = maze.getStart()
    Q = deque()
    visited = [[0]* maze.rows for i in range(maze.cols)]
    visited[startX][startY] = 1

    Q.append([(startX,startY)])

    while Q:
        selected_path = Q.popleft()
        (curr_nodeX,curr_nodeY) = selected_path[-1]
        if maze.isObjective(curr_nodeX,curr_nodeY):
            return selected_path
        for neighbor in maze.getNeighbors(curr_nodeX,curr_nodeY):
            if visited[neighbor[0]][neighbor[1]] == 0:
                visited[neighbor[0]][neighbor[1]] = 1
                new_path = selected_path + [neighbor]
                Q.append(new_path)
        
    return []


def astar(maze:Maze):
    start = maze.getStart()
    objective = maze.getObjectives()[0]  # Assuming there's only one objective

    # Function to calculate Manhattan distance
    def heuristic(cell, goal):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, objective), 0, start))  # (f, g, position)
    closed_list = set()
    came_from = {}
    g_score = {start: 0}

    while open_list:
        current_f, current_g, current = heapq.heappop(open_list)
        closed_list.add(current)
        if current == objective:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)  # optional
            return path[::-1]

        for neighbor in maze.getNeighbors(current[0], current[1]):
            if neighbor in closed_list:
                continue
            temp_g_score = g_score[current] + 1  # Assume cost between neighbors is 1
            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score = temp_g_score + heuristic(neighbor, objective)
                heapq.heappush(open_list, (f_score, temp_g_score, neighbor))

    return []  # if no path is found

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
    def __init__(self,pos, g, h, objs, p = None):  
        self.g = g
        self.h = h
        self.f = g + h
        self.pos = pos
        self.objs = objs
        self.p = p 

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
            if self.heap_list[i].f < self.heap_list[i // 2].f:
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
            if self.heap_list[i].f > self.heap_list[mc].f:
                temp = self.heap_list[i]
                self.heap_list[i] = self.heap_list[mc]
                self.heap_list[mc] = temp
            i = mc

    def min_child(self, i):
        if i * 2 + 1 > self.current_size:
            return i * 2
        else:
            if self.heap_list[i * 2].f < self.heap_list[i * 2 + 1].f:
                return i * 2
            else:
                return i * 2 + 1   
   

def astar_corner(maze:Maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    def astar_multi_heuristics(cell,objs):
        return minCostConnectPoints([cell,*objs])
    init = Node(
        maze.getStart(), 
        0, 
        astar_multi_heuristics(maze.getStart(), maze.getObjectives()), 
        maze.getObjectives()
    )
    init.p = init
    open_list = []
    frontier = BinaryHeap()
    came_from = {}
    closed_list = set()
    frontier.insert(init)
    while frontier.current_size > 0:     
        curr = frontier.del_min()
        if len(curr.objs) == 0:
            path = []
            path.insert(0, curr.pos)
            
            while curr.p is not curr:
                curr = curr.p
                path.insert(0, curr.pos)
            return path
        
        if frozenset([curr.pos,*curr.objs]) not in closed_list:
            closed_list.add(frozenset([curr.pos,*curr.objs]))
            for _,neighbor in enumerate(maze.getNeighbors(*curr.pos)):
                # temp_g_score = g_score[current] + 1  # Assume cost between neighbors is 1
                if neighbor in curr.objs:
                    new_objs = copy.deepcopy(curr.objs)
                    new_objs.remove(neighbor)
                else:
                    new_objs = copy.deepcopy(curr.objs)

                node = Node(
                    neighbor, 
                    curr.g+1, 
                    astar_multi_heuristics(neighbor, new_objs),  
                    new_objs, 
                    curr
                )
                frontier.insert(node)
    return []


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return astar_corner(maze)




def fast(maze):
    def manhattan_distance(point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def heuristic(current_pos, objectives):
        # Return distance to the closest objective
        if not objectives:
            return float('inf')  # Large value if no objectives left
        return min(manhattan_distance(current_pos, obj) for obj in objectives)

    def find_path_bfs(start, goal):
        """Find path from start to goal using BFS, return path."""
        if start == goal:
            return [start]
        queue = deque([([start], start)])
        visited = set([start])
        while queue:
            path, current = queue.popleft()
            for neighbor in maze.getNeighbors(*current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    if neighbor == goal:
                        return new_path
                    queue.append((new_path, neighbor))
        return []

    def greedy_best_first_search(start, objectives):
        objectives = set(objectives)  # Ensure objectives is a mutable set
        path = []
        current_pos = start
        
        while objectives:
            closest_obj = min(objectives, key=lambda obj: manhattan_distance(current_pos, obj))
            path_to_closest = find_path_bfs(current_pos, closest_obj)
            if not path_to_closest:
                return None  # Path not found to an objective

            if not path:
                path = path_to_closest
            else:
                path += path_to_closest[1:]  # Avoid duplicating the current position
            
            current_pos = closest_obj
            objectives.remove(closest_obj)
        
        return path

    start = maze.getStart()
    objectives = list(maze.getObjectives())
    return greedy_best_first_search(start, objectives)


    
