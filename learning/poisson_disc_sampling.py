import time
import numpy as np
import itertools
import random
np.random.seed(seed = int(time.time()))

class GridSampler:

    def __init__(self, extent, r, centered = False, k =20):
        self.n = len(extent)
        assert self.n == 2, "Currently only support 2 dimensions"
        self.extent = np.array(extent)
        self.r = r

        self.centered = centered
        self.points = []

        self.grid = np.mgrid[self.r/2:self.extent[0]:self.r, self.r/2:self.extent[1]:self.r].reshape(self.n, -1).T
        bins = extent/r
        bins = bins.astype(int)
        self.tot = 0
        self.grid = []
        for y_ind in range(bins[1]):
            self.grid.append([])
            for x_ind in range(bins[0]):
                self.grid[-1].append(np.array([x_ind*self.r, y_ind*self.r]))
                self.tot += 1
        self.k = k

        self.sampled = set()
        self.first_sample = False
        self.adjacent = []
        for i in itertools.product((-2, -1, 0, 1, 2), repeat = self.n):
            if i[0] == 0 and i[1] == 0:
                continue
            self.adjacent.append(i)


    def make_samples(self, num = 1, filter = lambda x: True):
        points = []
        for i in range(num):
            point = self.sample()
            if point is None:
                continue
            if not filter(point):
                points.append(point)
        return points

    def sample(self):
        if len(self.sampled) == self.tot:
            return None

        if not self.first_sample:
            rand_inds = np.random.randint(0, len(self.grid)), np.random.randint(0, len(self.grid[0]))
            self.sampled.add(rand_inds)
            self.first_sample = True
            pt =  self.grid[rand_inds[0]][rand_inds[1]]
            if self.centered:
                return pt - self.extent/2
            return pt
        
        for _ in range(self.k):
            samp = random.choice(list(self.sampled)) 
            adj = random.choice(self.adjacent)
            shape = (len(self.grid), len(self.grid[0]))
            rand_inds = [0,0]
            for i in range(2):
                rand_inds[i] = samp[i] + adj[i]
                if rand_inds[i] >= shape[i]:
                    rand_inds[i] -= shape[i]
                elif rand_inds[i] < 0:
                    rand_inds[i] += shape[i]
            rand_inds = tuple(rand_inds)
            if rand_inds in self.sampled:
                continue
            self.sampled.add(rand_inds)
            pt =  self.grid[rand_inds[0]][rand_inds[1]]
            if self.centered:
                return pt - self.extent/2
            return pt

        return None



class PoissonSampler:
    """
    An implementation of:
    https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
    """

    def __init__(self, extent, r, k = 30, centered = False, clump = False):
        """
        extent: an n dimensional array of the extent of the domain
        r: min distance between samples 
        k: maximum number of samples before rejection
        """
        self.n = len(extent)
        self.extent = np.array(extent)
        self.r = r
        self.k = k


        # grid is n dimensional
        # each dimension has 
        # extent[i]/(r/sqrt(n)) slots
        self.centered = centered
        bin_width = r/np.sqrt(self.n)
        shape = np.ceil(extent/bin_width).astype(np.int)
        self.bin_width = extent/shape
        self.grid = (-1*np.ones(shape)).astype(np.int)
        self.first_sample = False
        self.active_list = []
        self.points = []
        self.clump = clump

    def reset(self):
        self.grid = (-1*np.ones(self.grid.shape)).astype(np.int)
        self.first_sample = False
        self.active_list = []
        self.points = []

    def point_to_index(self, point):
        """
        point: an n dimensinoal numpy array point

        output: the row, col index of that point in self.grid as a tuple
        """
        if np.any(point > self.extent) or np.any(point < 0):
            return None
        return tuple(np.clip(np.floor(point/self.bin_width).astype(np.int), 0, self.grid.shape))

    def generate_annulus_point(self, center_point):
        """
        Generates a random point in an n-dimensional spherical
        annulus centered about `center_point` between r and 2r
        """
        dir = np.random.uniform(-1*np.ones(self.n), np.ones(self.n))
        if self.clump:
            mul = self.r
        else:
            mul = np.random.uniform(self.r, 2*self.r)
        dir = (dir/np.linalg.norm(dir)) * mul
        return dir + center_point

    def get_adjacent(self, point):
        """
        Returns the list of points indices in adjacent grid cells to
        point 
        """
        center_inds = np.array(self.point_to_index(point))
        assert center_inds is not None, "This point is out of bounds"

        adj_inds = []
        for offset in itertools.product((-2, -1, 0, 1, 2), repeat = self.n):
            inds = center_inds - np.array(offset)
            if np.any(inds >= self.grid.shape) or np.any(inds < 0):
                continue
            adj_inds.append(self.grid[tuple(inds)])
        return adj_inds


    def is_colliding(self, point, update_if_safe = None):
        """
        returns if point is colliding with another point
        (closer than r away)
        update_if_safe: if provided, will update the index
        in grid if safe to `update_if_safe`
        """
        inds = self.point_to_index(point)
        if inds is None:
            return True
        adj_inds = self.get_adjacent(point)
        res = False
        for ind in adj_inds:
            if ind == -1:
                continue
            close_point = self.points[ind]
            if np.linalg.norm(close_point - point) < self.r:
                res = True
                break
        if not res and update_if_safe is not None:
            self.grid[inds] = update_if_safe
        return res

    def make_samples(self, num = 1, filter= lambda x: True, verbose = False):
        samples = []
        for i in range(num):
            point = self.sample()
            if point is None:
                return samples
            if not filter(point):
                samples.append(point)
            elif verbose:
                print(f"REJECTING {point}")
        return samples

    def sample(self):
        """
        Sample from the poisson disc distribution 
        """
        if not self.first_sample:
            point = np.random.uniform(np.zeros(self.n), self.extent)
            self.active_list.append(point)
            self.points.append(point)
            inds = self.point_to_index(point)
            self.grid[inds] = len(self.points) - 1
            self.first_sample = True
            if self.centered:
                return point - self.extent/2
            return point

        while len(self.active_list) > 0:
            center_ind = np.random.randint(0, len(self.active_list))
            center_point = self.active_list[center_ind]
            for i in range(self.k):
                rand_point = self.generate_annulus_point(center_point)
                if not self.is_colliding(rand_point, update_if_safe=len(self.points)):
                    self.active_list.append(rand_point)
                    self.points.append(rand_point)
                    if self.centered:
                        return rand_point - self.extent/2
                    return rand_point
            self.active_list.pop(center_ind)

        return None

    def sample_gen(self):
        while True:
            point = self.sample()
            if point is None:
                return
            yield point


