import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import random, randint
from queue import PriorityQueue
from copy import deepcopy
from time import time
from google.colab import drive
drive.mount('/content/gdrive')

# Perline noise code courtesy of tgroid on SO
def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)
    return lerp(x1,x2,v)

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

if __name__ == 'main':
    lin = np.linspace(0,5,100,endpoint=False)
    x,y = np.meshgrid(lin,lin)

    plt.imshow(perlin(x,y,seed=5),origin='upper')
    plt.show()

class Point():

	def __init__(self, posx, posy):
		self.x = posx
		self.y = posy
		self.comparator = math.inf

	def __lt__(self, other):
		return self.comparator < other.comparator

	def __gt__(self, other):
		return self.comparator > other.comparator

	def __eq__(self, other):
		return self.x == other.x and self.y == other.y
def scale(X):
	X = ((X+1)/2 * 255).astype(int)
	return X

class Map():

	def __init__(self, length, width, cost_function='exp',seed=None,
	             filename=None, start=None, goal=None):

		self.seed = seed
		if self.seed == None:
			# Randomly assign seed from 0 to 10k if not provided
			self.seed = randint(0,10000)
		self.length = length
		self.width = width
		self.generateTerrain(filename)
		self.explored = []
		self.explored_lookup = {}
		for i in range(self.width):
			for j in range(self.length):
				self.explored_lookup[str(i)+','+str(j)] = False
		if start == None:
			self.start = Point(int(self.width*0.5),int(self.length*0.5))
		else:
			self.start = Point(start[0], start[1])
		if goal == None:
			self.goal = Point(int((self.width-1)*0.9),int((self.length-1)*0.9))
		else:
			self.goal = Point(goal[0], goal[1])
		if cost_function == 'exp':
			self.cost_function = lambda h0, h1: math.pow(math.e,h1-h0)
		elif cost_function == 'div':
			self.cost_function = lambda h0, h1: h0/(h1+1)
    
	'''generateTerrain: modifes self.map to either be the specified file, or
	randomly generated from perlin noise.
	input:
	filename - str, string of the npy file to generate the map
	seed - int, integer for reproducibility of a particular map
	octaves - int parameter for perlin noise
	output:
	None, self.map modified'''
	def generateTerrain(self, filename=None):
		if filename is None:
			linx = np.linspace(0,5,self.width,endpoint=False)
			liny = np.linspace(0,5,self.length,endpoint=False)
			x,y = np.meshgrid(linx,liny)
			self.map = scale(perlin(x, y, seed=self.seed))

		else:
			self.map = np.load(filename)
			self.width = self.map.shape[0]
			self.length = self.map.shape[1]

	def interpolate(self, a0, a1, w):
		if (0.0 > w):
			return a0
		if (1.0 < w):
			return a1
		return (a1 - a0) * ((w * (w * 6.0 - 15.0) + 10.0) * w ** 3) + a0

	def calculatePathCost(self, path):
		prev = path[0]
		if self.start != prev:
			print('Path does not start at start. Path starts at point: ' , str(prev.x),
				',', str(prev.y))
			return math.inf
		cost = 0
		for item in path[1:]:
			if self.isAdjacent(prev, item):
				cost += self.getCost(prev, item)
				prev = item
			else:
				print('Path does not connect at points: ', str(prev.x), ',', str(prev.y),
					' and ', str(item.x), ',', str(item.y))
				return math.inf
		if prev != self.goal:
			print('Path does not end at goal. Path ends at point: ' , str(prev.x),
				',', str(prev.y))
			return math.inf
		return cost

	def validTile(self, x, y):
		return x >= 0 and y >= 0 and x < self.width and y < self.length

	'''def validTile(self, p1):
		return self.validTile(p1.x, p1.y)'''

	def getTile(self, x, y):
		try:
			return self.map[x][y]
		except:
			print(x)
			print(y)
			print(self.length)
			print(self.width)
			raise()

	'''def getTile(self, p1):
		return self.getTile(p1.x, p1.y)'''

	def getCost(self, p1, p2):
		h0 = self.getTile(p1.x, p1.y)
		h1 = self.getTile(p2.x, p2.y)
		return self.cost_function(h0, h1)

	def isAdjacent(self, p1, p2):
		return (abs(p1.x - p2.x) == 1 or abs(p1.y - p2.y)) == 1 and (abs(p1.x - p2.x) < 2 and abs(p1.y - p2.y) < 2)

	def getNeighbors(self, p1):
		neighbors = []
		for i in [-1, 0, 1]:
			for j in [-1, 0, 1]:
				if i == 0 and j == 0:
					continue
				possible_point = Point(p1.x + i, p1.y + j)
				if self.validTile(possible_point.x, possible_point.y):
					neighbors.append(possible_point)
					if not self.explored_lookup[str(possible_point.x)+','+str(possible_point.y)]:
						self.explored_lookup[str(possible_point.x)+','+str(possible_point.y)] = True
						self.explored.append(possible_point)
		return neighbors

	def getStartPoint(self):
		return self.start

	def getEndPoint(self):
		return self.goal

	def getHeight(self):
		return np.amax(self.map)

	'''Creates a 2D image of the path taken and nodes explroed, prints
	pathcost and number of nodes explored'''
	def createImage(self, path):
		img = self.map
		path_img = np.zeros_like(self.map)
		explored_img = np.zeros_like(self.map)
		for item in self.explored:
			explored_img[item.x, item.y] = 1
		path_img_x = [item.x for item in path]
		path_img_y = [item.y for item in path]
		print('Path cost:', self.calculatePathCost(path))
		cmap = mpl.colors.ListedColormap(['white', 'red'])
		print('Nodes explored: ', len(self.explored) + len(path))
		plt.imshow(img, cmap='gray')
		plt.imshow(explored_img, cmap=cmap, alpha=0.3)
		plt.plot(path_img_y, path_img_x, linewidth=1)
		plt.show()

	'''Set the start and goal point on the 2D map, each point is a pair of integers'''
	def setStartGoal(self, start, goal):
		self.start = Point(np.clip(start[0], 0, self.length-1), np.clip(start[1], 0, self.width-1))
		self.goal = Point(np.clip(goal[0], 0, self.length-1), np.clip(goal[1], 0, self.width-1))
class Dijkstras(AIModule):

	def createPath(self, map_):
		q = PriorityQueue()
		''' Maintain three dictionaries to keep track of cost ("x,y" -> cost per
		node), previous (node -> parent). This keeps track of paths, and explored
		which helps us run faster by ignoring nodes already visited'''
		cost = {}
		prev = {}
		explored = {}
		# Dictionary initialization
		for i in range(map_.width):
			for j in range(map_.length):
				cost[(i,j)] = math.inf
				prev[(i,j)] = None
				explored[(i,j)] = False
		current_point = deepcopy(map_.start)
		current_point.comparator = 0 
		cost[(current_point.x, current_point.y)] = 0
		# Add start node to the queue
		q.put(current_point)
		# Search loop
		while q.qsize() > 0:
			# Get new point from PQ
			v = q.get()
			if explored[(v.x,v.y)]:
				continue
			explored[(v.x,v.y)] = True
			# Check if popping off goal
			if v == map_.getEndPoint():
				break
			# Evaluate neighbors
			neighbors = map_.getNeighbors(v)
			for neighbor in neighbors:
				alt = map_.getCost(v, neighbor) + cost[(v.x,v.y)]
				if alt < cost[(neighbor.x,neighbor.y)]:
					cost[(neighbor.x,neighbor.y)] = alt
					neighbor.comparator = alt
					prev[(neighbor.x,neighbor.y)] = v
				q.put(neighbor)
		# Find and return path
		path = []
		while v != map_.getStartPoint():
			path.append(v)
			v = prev[(v.x,v.y)]
		path.append(map_.getStartPoint())
		path.reverse()
		return path
 class AStarExp(AIModule):
    def createPath(self, map_):
        q = PriorityQueue()
        cost = {}
        prev = {}
        explored = {}
        # Dictionary initialization
        for i in range(map_.width):
            for j in range(map_.length):
                cost[(i,j)] = math.inf
                prev[(i,j)] = None
                explored[(i,j)] = False
        current_point = deepcopy(map_.start)
        endpoint = map_.getEndPoint()
        endpointh = map_.getTile(endpoint.x,endpoint.y)
        currenth = map_.getTile(current_point.x,current_point.y)
        current_point.comparator = self.heuristic(currenth, endpointh,current_point,endpoint)
        cost[(current_point.x, current_point.y)] = 0
        # Add start node to the queue
        q.put(current_point)
        # Search loop
        while q.qsize() > 0:
            # Get new point from PQ
            v = q.get()
            if explored[(v.x,v.y)]:
                continue
            explored[(v.x,v.y)] = True
            # Check if popping off goal
            if v == map_.getEndPoint():
                break
            # Evaluate neighbors
            neighbors = map_.getNeighbors(v)
            for neighbor in neighbors:
                # Use heuristic function to estimate remaining distance to goal
                neighborH = map_.getTile(neighbor.x,neighbor.y)
                h = self.heuristic(neighborH, endpointh,neighbor,endpoint)

                # Update cost and comparator based on heuristic and current cost
                alt = map_.getCost(v, neighbor) + cost[(v.x,v.y)]
                if alt < cost[(neighbor.x,neighbor.y)]:
                    cost[(neighbor.x,neighbor.y)] = alt
                    neighbor.comparator = alt + h
                    prev[(neighbor.x,neighbor.y)] = v
                q.put(neighbor)
        # Find and return path
        path = []
        while v != map_.getStartPoint():
            path.append(v)
            v = prev[(v.x,v.y)]
        path.append(map_.getStartPoint())
        path.reverse()
        return path
    
    def heuristic(self, current, goal,x,y):
        height_diff = goal - current
        vertical_dis = y.y - x.y
        horizontal_dis = y.x -x.x
        if vertical_dis > horizontal_dis:
          if height_diff > 0:
            return  math.exp(1)*height_diff - (goal - current) + vertical_dis
          elif height_diff == 0:
            return vertical_dis
          else:
            return math.exp(-1)*height_diff - (current - goal) + vertical_dis
        else:
          if height_diff > 0:
            return  math.exp(1)*height_diff - (goal - current) + horizontal_dis
          elif height_diff == 0:
            return horizontal_dis
          else:
            return math.exp(-1)*height_diff - (current - goal) + horizontal_dis
class AStarDiv(AIModule):
    def createPath(self, map_):
        q = PriorityQueue()
        cost = {}
        prev = {}
        explored = {}
        # Dictionary initialization
        for i in range(map_.width):
            for j in range(map_.length):
                cost[(i,j)] = math.inf
                prev[(i,j)] = None
                explored[(i,j)] = False
        current_point = deepcopy(map_.start)
        endpoint = map_.getEndPoint()
        endpointh = map_.getTile(endpoint.x,endpoint.y)
        currenth = map_.getTile(current_point.x,current_point.y)
        current_point.comparator = self.heuristic2(currenth, endpointh,current_point,endpoint)
        cost[(current_point.x, current_point.y)] = 0
        # Add start node to the queue
        q.put(current_point)
        # Search loop
        while q.qsize() > 0:
            # Get new point from PQ
            v = q.get()
            if explored[(v.x,v.y)]:
                continue
            explored[(v.x,v.y)] = True
            # Check if popping off goal
            if v == map_.getEndPoint():
                break
            # Evaluate neighbors
            neighbors = map_.getNeighbors(v)
            for neighbor in neighbors:
                # Use heuristic function to estimate remaining distance to goal
                neighborH = map_.getTile(neighbor.x,neighbor.y)
                h = self.heuristic2(neighborH, endpointh,neighbor,endpoint)

                # Update cost and comparator based on heuristic and current cost
                alt = map_.getCost(v, neighbor) + cost[(v.x,v.y)]
                if alt < cost[(neighbor.x,neighbor.y)]:
                    cost[(neighbor.x,neighbor.y)] = alt
                    neighbor.comparator = alt + h
                    prev[(neighbor.x,neighbor.y)] = v
                q.put(neighbor)
        # Find and return path
        path = []
        while v != map_.getStartPoint():
            path.append(v)
            v = prev[(v.x,v.y)]
        path.append(map_.getStartPoint())
        path.reverse()
        return path
    
    def heuristic2(self, current, goal,x,y):
        height_diff = goal - current
        vertical_dis = y.y - x.y
        horizontal_dis = y.x -x.x
        if vertical_dis > horizontal_dis:
          return (0.40* vertical_dis) 
        else:
          return (0.40* horizontal_dis) 
          

class AStarMSH(AIModule):
  def createPath(self, map_):
        q = PriorityQueue()
        cost = {}
        prev = {}
        explored = {}
        # Dictionary initialization
        for i in range(map_.width):
            for j in range(map_.length):
                cost[(i,j)] = math.inf
                prev[(i,j)] = None
                explored[(i,j)] = False
        current_point = deepcopy(map_.start)
        endpoint = map_.getEndPoint()
        endpointh = map_.getTile(endpoint.x,endpoint.y)
        currenth = map_.getTile(current_point.x,current_point.y)
        current_point.comparator = self.heuristic3(currenth, endpointh,current_point,endpoint)
        cost[(current_point.x, current_point.y)] = 0
        # Add start node to the queue
        q.put(current_point)
        # Search loop
        while q.qsize() > 0:
            # Get new point from PQ
            v = q.get()
            if explored[(v.x,v.y)]:
                continue
            explored[(v.x,v.y)] = True
            # Check if popping off goal
            if v == map_.getEndPoint():
                break
            # Evaluate neighbors
            neighbors = map_.getNeighbors(v)
            for neighbor in neighbors:
                # Use heuristic function to estimate remaining distance to goal
                neighborH = map_.getTile(neighbor.x,neighbor.y)
                h = self.heuristic3(neighborH, endpointh,neighbor,endpoint)
                c = 1.1
                h = h * c

                # Update cost and comparator based on heuristic and current cost
                alt = map_.getCost(v, neighbor) + cost[(v.x,v.y)]
                if alt < cost[(neighbor.x,neighbor.y)]:
                    cost[(neighbor.x,neighbor.y)] = alt
                    neighbor.comparator = alt + h
                    prev[(neighbor.x,neighbor.y)] = v
                q.put(neighbor)
        # Find and return path
        path = []
        while v != map_.getStartPoint():
            path.append(v)
            v = prev[(v.x,v.y)]
        path.append(map_.getStartPoint())
        path.reverse()
        return path

    
  def heuristic3(self, current, goal,x,y):
        height_diff = goal - current
        vertical_dis = y.y - x.y
        horizontal_dis = y.x -x.x
        if vertical_dis > horizontal_dis:
          if height_diff > 0:
            return  math.exp(1)*height_diff - (goal - current) + vertical_dis
          elif height_diff == 0:
            return vertical_dis
          else:
            return math.exp(-1)*height_diff - (current - goal) + vertical_dis
        else:
          if height_diff > 0:
            return  math.exp(1)*height_diff - (goal - current) + horizontal_dis
          elif height_diff == 0:
            return horizontal_dis
          else:
            return math.exp(-1)*height_diff - (current - goal) + horizontal_dis

#Testing          
cost_function = 'exp'
#cost_function = 'div'
w = 500
l = 500
start = None
goal = None
seed = 0

AI = AStarExp()
#AI = AStarMSH()

# Change to the filepath on your drive
filename = "/content/gdrive/My Drive/Colab Notebooks/msh.npy"

m = Map(w,l, seed=seed, filename=filename, start=start, goal=goal)
t1 = time()
path = AI.createPath(m)
t2 = time()
print('Time (s): ', t2-t1)
m.createImage(path)
