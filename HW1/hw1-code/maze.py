# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import re
import copy
from collections import Counter

class Maze:
    # Initializes the Maze object by reading the maze from a file
    def __init__(self, filename):
        self.__filename = filename
        self.__wallChar = '%'
        self.__startChar = 'P'
        self.__objectiveChar = '.'
        self.__start = None
        self.__objective = []
        self.__states_explored = 0

        with open(filename) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = [list(line.strip('\n')) for line in lines]

        self.rows = len(lines)
        self.cols = len(lines[0])
        self.mazeRaw = lines

        if (len(self.mazeRaw) != self.rows) or (len(self.mazeRaw[0]) != self.cols):
            print("Maze dimensions incorrect")
            raise SystemExit
            return

        for row in range(len(self.mazeRaw)):
            for col in range(len(self.mazeRaw[0])):
                if self.mazeRaw[row][col] == self.__startChar:
                    self.__start = (row, col)
                elif self.mazeRaw[row][col] == self.__objectiveChar:
                    self.__objective.append((row, col))

    # Returns True if the given position is the location of a wall
    def isWall(self, row, col):
        return self.mazeRaw[row][col] == self.__wallChar

    # Rturns True if the given position is the location of an objective
    def isObjective(self, row, col):
        return (row, col) in self.__objective

    # Returns the start position as a tuple of (row, column)
    def getStart(self):
        return self.__start

    def setStart(self, start):
        self.__start = start

    # Returns the dimensions of the maze as a (row, column) tuple
    def getDimensions(self):
        return (self.rows, self.cols)

    # Returns the list of objective positions of the maze
    def getObjectives(self):
        return copy.deepcopy(self.__objective)


    def setObjectives(self, objectives):
        self.__objective = objectives


    def getStatesExplored(self):
        return self.__states_explored

    # Check if the agent can move into a specific row and column
    def isValidMove(self, row, col):
        return row >= 0 and row < self.rows and col >= 0 and col < self.cols and not self.isWall(row, col)

    # Returns list of neighboring squares that can be moved to from the given row,col
    def getNeighbors(self, row, col):
        possibleNeighbors = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ]
        neighbors = []
        for r, c in possibleNeighbors:
            if self.isValidMove(r,c):
                neighbors.append((r,c))
        self.__states_explored += 1
        return neighbors

    def isValidPath(self, path):
        # check if path is in correct shape (type, not empty)
        if not isinstance(path, list):
            return "path must be list"

        if len(path) == 0:
            return "path must not be empty"

        if not isinstance(path[0], tuple):
            return "position must be tuple"

        if len(path[0]) != 2:
            return "position must be (x, y)"

        # check single hop
        for i in range(1, len(path)):
            prev = path[i-1]
            cur = path[i]
            dist = abs((prev[1]-cur[1])+(prev[0]-cur[0]))
            if dist > 1:
                return "Not single hop"

        # check whether it is valid move
        for pos in path:
            if not self.isValidMove(pos[0], pos[1]):
                return "Not valid move"

        # check whether it passes all goals
        if not set(self.__objective).issubset(set(path)):
            return "Not all goals passed"

        # check whether it ends up at one of goals
        if not path[-1] in self.__objective:
            return "Last position is not goal"

        # check for duplication
        if len(set(path)) != len(path):
            c = Counter(path)
            dup_dots = [p for p in set(c.elements()) if c[p] >= 2]
            for p in dup_dots:
                indices = [i for i, dot in enumerate(path) if dot == p]
                is_dup = True
                for i in range(len(indices) - 1):
                    for dot in path[indices[i]+1: indices[i + 1]]:
                        if self.isObjective(dot[0], dot[1]):
                            is_dup = False
                            break
                if is_dup:
                    return "Unnecessary path detected"
        return "Valid"
