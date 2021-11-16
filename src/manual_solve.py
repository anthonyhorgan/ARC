#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
from itertools import product
import torch
from torch import nn
from matplotlib import pyplot as plt

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

# Student Name:     Anthony Horgan
# Student ID:       17452572
# Github repo url:  https://github.com/anthonyhorgan/ARC.git

class Conv2d:
    def __init__(self, kernel_shape, kernel_value=None):
        self.height = kernel_shape[0]
        self.width = kernel_shape[1]
        if kernel_value is None:
            self.kernel = np.random.rand(kernel_shape)
        else:
            self.kernel = kernel_value

    def __call__(self, x):
        output_height = x.shape[0] - self.height + 1
        output_width = x.shape[1] - self.width + 1
        ret_x = np.zeros((output_height, output_width))
        for i in range(0, output_height):
            for j in range(0, output_width):
                ret_x[i, j] = np.sum(self.kernel * x[i:i + self.height, j:j + self.width])

        return ret_x


def solve_6e19193c(x):
    colour_number = np.max(x) # get colour to paint with. only one colour used per example (other than black)
    ret_x = np.copy(x)
    # sources of projectile
    # pad left right top bottom
    sources = [
        {"conv": Conv2d((2, 2), np.array([[1, 1], [1, 0]])), "pad_width": ((1, 0), (1, 0)), "direction": np.array([1, 1])},
        {"conv": Conv2d((2, 2), np.array([[1, 1], [0, 1]])), "pad_width": ((1, 0), (0, 1)), "direction": np.array([1, -1])},
        {"conv": Conv2d((2, 2), np.array([[0, 1], [1, 1]])), "pad_width": ((0, 1), (0, 1)), "direction": np.array([-1, -1])},
        {"conv": Conv2d((2, 2), np.array([[1, 0], [1, 1]])), "pad_width": ((0, 1), (1, 0)), "direction": np.array([-1, 1])},
    ]

    projectiles = []
    for source in sources:
        conv = source["conv"]
        pad_width = source["pad_width"]
        direction = source["direction"]
        conv_out = conv(ret_x)
        padded_conv_out = np.pad(conv_out, pad_width, mode="constant", constant_values=((0, 0), (0, 0)))
        # TODO currently only handling case where there is at most one source of any given type
        center_idx = np.where(padded_conv_out == 3 * colour_number)
        if center_idx[0].shape == (0,):
            continue
        projectiles.append({"center": np.array([int(center_idx[0]), int(center_idx[1])]), "direction": direction})

    # #TODO see about indexing np arrays with np arrays
    for projectile in projectiles:
        # center is the cats cradle
        curr_idx = projectile["center"]
        direction = projectile["direction"]
        while True:
            try:
                curr_idx += direction
                #TODO make this more numpy-ey
                if any(curr_idx < 0):
                    break
                ret_x[curr_idx[0], curr_idx[1]] = colour_number
            except IndexError:
                break

    return ret_x


def solve_28e73c20(x):
    # spiral
    ret_x = np.ones_like(x) * 3
    ret_x = np.pad(ret_x, ((1, 1), (1, 1)), "constant", constant_values=((0, 0), (0, 0)))
    start_x = 1
    start_y = 2
    command_mode_dict = {"east": {"ahead": np.array([0, 1]), "right": np.array([1, 0])},
                         "south": {"ahead": np.array([1, 0]), "right": np.array([0, -1])},
                         "west": {"ahead": np.array([0, -1]), "right": np.array([-1, 0])},
                         "north": {"ahead": np.array([-1, 0]), "right": np.array([0, 1])}
                         }

    start_idx = np.array([start_y, start_x])
    curr_idx = start_idx
    at_least_one_step_taken = True
    while at_least_one_step_taken:
        at_least_one_step_taken = False
        for mode, mode_dict in command_mode_dict.items():
            while True:
                ret_x[curr_idx[0], curr_idx[1]] = 0     # colour current square black
                ahead = curr_idx + mode_dict["ahead"]
                two_ahead_idx = curr_idx + 2 * mode_dict["ahead"]
                ahead_right_idx = curr_idx + mode_dict["ahead"] + mode_dict["right"]
                if ret_x[two_ahead_idx[0], two_ahead_idx[1]] == 0 or ret_x[ahead_right_idx[0], ahead_right_idx[1]] == 0 or ret_x[ahead[0], ahead[1]] == 0:
                    break
                at_least_one_step_taken = True
                curr_idx = curr_idx + mode_dict["ahead"]

    ret_x = ret_x[1:-1, 1:-1] # remove padding
    return ret_x


def fill(a, pos, colour):
    height, width = a.shape
    y, x = pos
    a[y, x] = colour
    # check east
    if x + 1 < width and not a[y, x + 1]:
        fill(a, (y, x + 1), colour)
    # check south
    if y + 1 < height and not a[y + 1, x]:
        fill(a, (y + 1, x), colour)
    # check west
    if x - 1 >= 0 and not a[y, x - 1]:
        fill(a, (y, x - 1), colour)
    # check north
    if y - 1 >= 0 and not a[y - 1, x]:
        fill(a, (y - 1, x), colour)


def solve_83302e8f(x):
    ret_x = np.copy(x)
    garden_colour = 3
    path_colour = 4

    x_grid = np.zeros_like(x)
    # print(a[0])
    grid_colour = np.max(x)
    height, width = x.shape
    for i, j in zip(*np.where(x)):
        if j == 0 or j == width - 1:
            x_grid[:, i] = grid_colour
        if i == 0 or i == height - 1:
            x_grid[j, :] = grid_colour

    starting_points = np.where(x != x_grid)
    for starting_point in zip(*starting_points):
        fill(ret_x, starting_point, path_colour)
    ret_x[np.where(ret_x == 0)] = garden_colour
    # print(ret_x)
    # plt.figure()
    # plt.imshow(ret_x)
    # plt.show()
    # exit()
    return ret_x


def solve_90c28cc7(x):
    x = np.copy(x)

    # remove black squares from input
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

    # get shape
    x_height, x_width = x.shape

    vert_idxs = [0]
    ref_slice = x[0]
    for i in range(1, x_height):
        slice = x[i]
        if not np.all(slice == ref_slice):
            ref_slice = slice
            vert_idxs.append(i)

    horiz_idxs = [0]
    ref_slice = x[:, 0]
    for j in range(1, x_width):
        slice = x[:, j]
        if not np.all(slice == ref_slice):
            ref_slice = slice
            horiz_idxs.append(j)

    ret_x = np.zeros((len(vert_idxs), len(horiz_idxs)), dtype=np.int32)
    # TODO change variable names
    for (big_i, big_j), (small_i, small_j) in zip(product(vert_idxs, horiz_idxs), product(range(len(vert_idxs)), range(len(horiz_idxs)))):
        # print(big_i, big_j, "     ", small_i, small_j)
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

