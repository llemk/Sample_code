#!/usr/bin/env python3
import numpy as np
import random
import math


class Node:
    def __init__(self, state, parent=None, path_cost=0, h_value=None, visited_neighbors=[]):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.h_value = h_value
        self.visited_neighbors = visited_neighbors

    def queen_check(self):
        state = self.state
        # initialize counter for number of queens in the same row and same diagonal
        same_row = 0
        same_dia = 0
        # for each column
        for i in range(len(state)):
            # find the row the queen is in
            queen_row = state[i]
            queen_row_val = int(queen_row)
            # check how many queens are in the same row
            same_row += state.count(queen_row) - 1
            # check how many queens are on the same diagonal
            for j in range(len(state)-1):
                column = (i + j + 1) % 8
                dist = abs(i - column)
                if int(state[column]) == queen_row_val + dist or int(state[column]) == queen_row_val - dist:
                    same_dia += 1
        # divide the total number of queens interfering by 2 because each interference is counted twice
        total_interference = (same_row + same_dia) // 2
        # update the nodes value
        self.h_value = total_interference
        return total_interference

    def get_neighbors(self):
        input_state = self.state
        # initialize an empty list of neighbors
        neighbors = []
        for i in range(56):
            # column of the queen to be moved
            queen_loc = i // 7
            state_copy = list(input_state)
            # row of the queen to be moved
            queen_value = int(input_state[queen_loc])
            # increment the queen row to a new value between 0 and 7
            new_queen_value = (queen_value + (i % 7) + 1) % 8
            # insert the new queen row into the state
            state_copy[queen_loc] = str(new_queen_value)
            new_state = ''.join(state_copy)
            # create a new node for the new state
            new_node = Node(new_state, self, self.path_cost+1)
            new_node.queen_check()
            # add the new node to the list of neighbors
            neighbors.append(new_node)
        # sort the neighbors by the number of interfering queens
        neighbors.sort(key=lambda neighbors: neighbors.h_value)
        neighbors.reverse
        return neighbors

    def random_neighbor(self):
        # if all neighbors visited then return nothing
        if len(self.visited_neighbors) == 56:
            return

        # initialize the open neighbors to be visited
        open_n = [*range(0, 56, 1)]
        # remove neighbors that have already been visited
        open_n = [i for i in open_n if i not in self.visited_neighbors]
        # select a random neighbor to be visited (int between 0 and 55)
        r = open_n[random.randint(0, len(open_n)-1)]
        self.visited_neighbors.append(r)
        # find the column of the queen to be moved
        queen_loc = r // 7
        state_copy = list(self.state)
        queen_value = int(state_copy[queen_loc])
        # find the new row value of the queen between 0 and 7 but not including the value of the current location
        new_queen_value = (queen_value + (r % 7) + 1) % 8
        # insert the new queen row value into the state
        state_copy[queen_loc] = str(new_queen_value)
        new_state = ''.join(state_copy)
        # initialize the new node for the new state
        new_node = Node(new_state, self, self.path_cost + 1)
        new_node.queen_check()
        # set the new nodes visited neighbors to be empty
        new_node.visited_neighbors = []
        return new_node


def hillclimb_sa(state):
    # initialize a node with the input state
    current_node = Node(state)
    current_node.queen_check()
    current_h = current_node.h_value
    # get all neighbors of the input node
    neighbors = current_node.get_neighbors()
    # select the neighbor with the lowest number of interfering queens
    test_node = neighbors[0]
    child_h = test_node.h_value
    # continue the loop until the number of interfering queens in the child is no long decreasing
    while current_h > child_h:
        # set the child to be the current node
        current_node = test_node
        current_h = test_node.h_value
        # get the neighbors of the new current node
        neighbors = current_node.get_neighbors()
        # get the new child node
        test_node = neighbors[0]
        child_h = test_node.h_value

    # if the goal state was never found return a failed success indicator and 0 for step length
    if current_node.h_value > 0:
        return 0, 0
    # if the goal state was found, return a success and step length
    elif current_node.h_value == 0:
        return 1, current_node.path_cost
    return


def hillclimb_fc(state):
    # initialize the input node
    node = Node(state)
    node.queen_check()
    node.visited_neighbors = []
    # infinite loop
    while 1:
        # set the next node to be a random neighbor
        next_n = node.random_neighbor()
        # if there are no more neighbors return failure
        if next_n is None:
            return 0, 0
        # while the child node has less interfering queens than the parent
        while next_n.h_value < node.h_value:
            # set the child node to be the new current node
            node = next_n
            next_n = node.random_neighbor()
            # if the goal state is found return success and number of steps
            if node.h_value == 0:
                return 1, node.path_cost
    return


def sim_anneal(state):
    # initialize the input node
    node = Node(state)
    node.queen_check()
    node.visited_neighbors = []
    for t in range(1000):
        # if the goal state is found return success and number of steps
        if node.h_value == 0:
            return 1, node.path_cost
        # at the last time step set temperature to zero
        if t == 999:
            temp = 0
        else:
            # calculate the temperature from the schedule
            temp = 100*math.exp(-0.5*t)
        # if the temperature reaches zero
        if temp == 0:
            # if the goal state is not found return failure
            if node.h_value > 0:
                return 0, 0
            # if the goal state is found return success and number of steps
            elif node.h_value == 0:
                return 1, node.path_cost
        # get a random neighbor
        next_node = node.random_neighbor()
        # if there are no more neighbors return failure
        if next_node is None:
            return 0, 0
        # calculate delta E
        del_e = node.h_value - next_node.h_value
        # if the next node has a lower number of interfering queens
        if del_e >= 0:
            node = next_node
        # allow for bad moves to be allowed randomly
        elif random.random() <= math.exp(del_e/temp):
            node = next_node
    return


if __name__ == '__main__':

    num_states = 1000
    # generate an array that holds 1000 random states
    rand_init_states = np.random.randint(0, 8, num_states + 7).astype(str)

    # initalize empty list for results of each algorithm
    sa_result = []
    fc_result = []
    sim_result = []

    for ii in range(num_states):
        # increment through the random states
        init_state = rand_init_states[ii: ii+8]
        init_state = ''.join(init_state)
        # run both hill climbs and simulated annealing on the random state
        sa_result.append(hillclimb_sa(init_state))
        fc_result.append(hillclimb_fc(init_state))
        sim_result.append(sim_anneal(init_state))

    # initialize counters to zero
    sa_win = 0
    sa_steps = 0
    fc_win = 0
    fc_steps = 0
    sim_win = 0
    sim_steps = 0

    # for each result
    for jj in range(num_states):
        # get the success and number of steps from each result
        aa = sa_result[jj]
        bb = fc_result[jj]
        cc = sim_result[jj]
        # add the success and number of steps to get the total number of successful runs and the total steps
        sa_win += aa[0]
        sa_steps += aa[1]
        fc_win += bb[0]
        fc_steps += bb[1]
        sim_win += cc[0]
        sim_steps += cc[1]

    e = 0.000001    # constant to avoid division by zero

    # calculate the average number of steps to solve for each algorithm
    sa_avg = sa_steps/(sa_win + e)
    fc_avg = fc_steps/(fc_win + e)
    sim_avg = sim_steps/(sim_win + e)

    # print results
    print('STEEPA % success: ', sa_win/num_states)
    print('STEEPA avg steps: ', sa_avg)
    print('FC % success: ', fc_win/num_states)
    print('FC avg steps: ', fc_avg)
    print('SIMA % success: ', sim_win/num_states)
    print('SIMA avg steps: ', sim_avg)