#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
from itertools import product

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

# Student Name:     Anthony Horgan
# Student ID:       17452572
# Github repo url:  https://github.com/anthonyhorgan/ARC.git

# problems solved:
#   6e19193c - "projectiles"
#   83302e8f - "paths and gardens"
#   90c28cc7 - "shrink regions"
#
# Summary/Reflection
# Numpy is the main tool that is used to solve the problems. Seeing as the input and output was specified as np arrays
# it seemed logical to use the many useful features of numpy to manipulate the arrays
# np.where proved to be a useful method when trying to identify important cells within the input
#
# My solution for "paths and gardens" and "projectiles" both had a common structure of identifying "objects" in the input (both functions
# use np.where) before drawing on cells as needed to produce the input. In both cases, the drawing consisted of
# drawing a colour on the current position in the grid, moving to the next grid position based on some rules and repeating.
#
# The only other python library function that used was itertools.product in "shrink regions". This could have been replaced
# by np.meshgrid but I felt itertools.product simpler/cleared
#
# A commonality between "shrink regions" and "projectiles" was looping over parts of the input array to identify patterns.
# In "projectile" I convolved a filter over the input to identify the "canons". In "shrink regions" I looped over the
# columns and rows to identify and remove black columns/rows and also to identify the boundaries of the regions
#
# I tried to chose problems that were different from each other which explains why my solutions don't share that much code


class Conv2d:
    '''
    This is a simplified version of a 2d convolution used in deep learning and image processing
    '''
    def __init__(self, kernel_value):
        '''
        kernel_value: list of lists or 2d array denoting the values the kernel should have
        '''
        self.kernel = kernel_value
        self.height = len(kernel_value)
        self.width = len(kernel_value[0])

    def __call__(self, x):
        # calculate output shape
        output_height = x.shape[0] - self.height + 1
        output_width = x.shape[1] - self.width + 1
        ret_x = np.zeros((output_height, output_width))
        # perform convolution operation
        for i in range(0, output_height):
            for j in range(0, output_width):
                ret_x[i, j] = np.sum(self.kernel * x[i:i + self.height, j:j + self.width])

        return ret_x


def solve_6e19193c(x):
    '''
    The input array contains a variable number of v shapes made up of 3 cells. I call these shapes cannons.
    The cannons can be oriented to face 4 different directions.
    SE ■■  SW ■■  NW  ■  NE■
       ■       ■     ■■    ■■
    Define the center of a cannon to be the cell which, if floodfilled in would transform the v shape into a square
    Then we must draw the path of a projectile fired from the center in the direction that the cannon is facing.
    Don't draw on the center itself

    All training and test grids are solved correctly
    '''
    # projectile path
    colour_number = np.max(x) # get colour to paint with. only one colour used per example (other than black)
    ret_x = np.copy(x)
    # Step 1. Identify the position and direction of each cannon in the input
    # for each of the 4 different cannon orientations, we define
    #   - a convolution with a kernel that will match the pattern of a cannon pointing in one of the 4 directions,
    #   - the padding to apply to the output of the convolution in order to identify the center of the cannon,
    #   - the direction of the cannon
    # The kernel size will be (2, 2) since we are matching a 2x2 pattern.
    # If the input to the convolution is (n, n) then the output will be (n-1, n-1) will have zero in each cell except
    # where the pattern was detected. The pad_width determines on which sides of the output to append a row/column of
    # zeros. This will move the non-zero cell to be in the same coordinates as the center of the cannon in the input
    cannons = [
        {"conv": Conv2d(np.array([[1, 1], [1, 0]])), "pad_width": ((1, 0), (1, 0)), "direction": np.array([1, 1])},
        {"conv": Conv2d(np.array([[1, 1], [0, 1]])), "pad_width": ((1, 0), (0, 1)), "direction": np.array([1, -1])},
        {"conv": Conv2d(np.array([[0, 1], [1, 1]])), "pad_width": ((0, 1), (0, 1)), "direction": np.array([-1, -1])},
        {"conv": Conv2d(np.array([[1, 0], [1, 1]])), "pad_width": ((0, 1), (1, 0)), "direction": np.array([-1, 1])},
    ]

    projectiles = []
    for cannon in cannons:
        # extract information from cannon dict
        conv = cannon["conv"]
        pad_width = cannon["pad_width"]
        direction = cannon["direction"]
        conv_out = conv(ret_x)  # feed the input through the convolution
        # pad the conv output to position non-zero cell(s) over the center of the cannon(s)
        padded_conv_out = np.pad(conv_out, pad_width, mode="constant", constant_values=((0, 0), (0, 0)))
        # A cell in the output will have a value of 3 * colour_number if the cells in the input which contributed to it
        # matche the pattern in the kernel
        center_idx = np.where(padded_conv_out == 3 * colour_number)
        if center_idx[0].shape == (0,):
            continue

        for y, x in zip(*center_idx):
            # create a projectile for each cannon which will travel from the cannons center in a direction
            projectiles.append({"center": np.array([int(y), int(x)]), "direction": direction})

    # Step 2: draw the path of each projectile
    for projectile in projectiles:
        curr_idx = projectile["center"]
        direction = projectile["direction"]
        while True:
            try:
                # move one step in direction
                curr_idx += direction
                if any(curr_idx < 0):
                    # break if y or x in current position is negative
                    break
                # draw on current position
                ret_x[curr_idx[0], curr_idx[1]] = colour_number
            except IndexError:
                # break if position is outside array
                break

    return ret_x


def flood_fill(a, pos, colour):
    '''
    Helper function used in solve_83302e8f
    flood floodfills black cells. The floodfill will be bounded by any cell other than black
    a: input array
    pos: current position to colour
    colour: which colour to paint with
    '''
    height, width = a.shape
    y, x = pos
    a[y, x] = colour    # colour in current position
    # If any of the cells beside the current cell (orthogonal directions) are black, call floodfill on that cell
    # check east
    if x + 1 < width and not a[y, x + 1]:
        flood_fill(a, (y, x + 1), colour)
    # check south
    if y + 1 < height and not a[y + 1, x]:
        flood_fill(a, (y + 1, x), colour)
    # check west
    if x - 1 >= 0 and not a[y, x - 1]:
        flood_fill(a, (y, x - 1), colour)
    # check north
    if y - 1 >= 0 and not a[y - 1, x]:
        flood_fill(a, (y - 1, x), colour)


def solve_83302e8f(x):
    '''
    The input array is divided up into equally-sized square regions.
    These regions are divided by walls. There are gaps in the wall which connect some regions. I will call these
    connected regions paths
    Some regions are completely surrounded by walls. I will call these regions gardens.
    To solve the problem, we need to colour all of the path regions yellow and colour all of the garden regions green.

    All training and test grids are solved correctly
    '''
    ret_x = np.copy(x)  # deep copy np array
    garden_colour = 3   # green
    path_colour = 4     # yellow

    x_grid = np.zeros_like(x)
    grid_colour = np.max(x)     # find out what colour the walls are. (the wall cells are the only non-black cells in the input)
    height, width = x.shape
    # Construct an array of "completed walls". i.e. floodfill in the gaps in the walls from the input array
    # We can deduce what the completed walls should look like by going around the cells at the edges
    #     (i.e. cells where x=0 or x=-1 or y=0 or y=-1), if a cell contains a wall, then floodfill in the row or column
    #     with walls
    for i, j in zip(*np.where(x)):
        if j == 0 or j == width - 1:
            x_grid[:, i] = grid_colour
        if i == 0 or i == height - 1:
            x_grid[j, :] = grid_colour

    # identify the gaps in the walls in the input array by comparing it with the completed walls array
    starting_points = np.where(x != x_grid)
    # set these "gaps" as starting points for the paths and flood fill in the paths from there
    for starting_point in zip(*starting_points):
        flood_fill(ret_x, starting_point, path_colour)
    # once paths are coloured in, every black cell is a garden, so colour it in green
    ret_x[np.where(ret_x == 0)] = garden_colour
    return ret_x


def solve_90c28cc7(x):
    '''
    Shrink Pattern
    The input array contains a rectangle made up of several regions.
    Each region is made up of several cells of the same colour
    The task is to represent each region as a single cell.
    The colour colour and position of each cell in the output should correspond to a region in the input.

    All training and test grids are solved correctly
    '''
    x = np.copy(x)

    # remove black squares from input
    # if all the cells in a row or column of x are black, then delete that row or column
    rows = []
    cols = []
    for i in range(x.shape[0]):
        if not np.any(x[i]):
            rows.append(i)
    for j in range(x.shape[1]):
        if not np.any(x[:, j]):
            cols.append(j)

    x = np.delete(x, rows, axis=0)
    x = np.delete(x, cols, axis=1)

    # get new shape (after removing black)
    x_height, x_width = x.shape

    # We need to find the region borders

    # find the horizontal region borders
    vert_idxs = [0]     # list to store the column indexes where a new region starts
    ref_slice = x[0]
    for i in range(1, x_height):
        slice = x[i]
        if not np.all(slice == ref_slice):
            ref_slice = slice
            vert_idxs.append(i)

    # find the horizontal region borders
    horiz_idxs = [0]    # list to store the row indexes where a new region starts
    ref_slice = x[:, 0]
    for j in range(1, x_width):
        slice = x[:, j]
        if not np.all(slice == ref_slice):
            ref_slice = slice
            horiz_idxs.append(j)

    ret_x = np.zeros((len(vert_idxs), len(horiz_idxs)), dtype=np.int32)
    big_idxs = product(vert_idxs, horiz_idxs)                           # coordinates of top left cell of each region
    small_idxs = product(range(len(vert_idxs)), range(len(horiz_idxs))) # coordinates of every cell in output

    # output cell colour = colour of top left cell of region corresponding to output cell
    for (big_i, big_j), (small_i, small_j) in zip(big_idxs, small_idxs):
        ret_x[small_i, small_j] = x[big_i, big_j]
    return ret_x



def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

