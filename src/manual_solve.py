#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


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

