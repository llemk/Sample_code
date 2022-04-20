import numpy as np
import time
import sys


def visualize(state):
    state = str(state)
    vis = np.array([[state[0], state[1], state[2]], [state[3], state[4], state[5]], [state[6], state[7], state[8]]])
    print(vis)
    return


def get_actions(state):
    # 3x3 puzzle arranged as [A B C]
    #                        [D E F]
    #                        [G H I]
    action_dic = {'A': ['left', 'up'],
                  'B': ['right', 'left', 'up'],
                  'C': ['right', 'up'],
                  'D': ['left', 'up', 'down'],
                  'E': ['right', 'left', 'up', 'down'],
                  'F': ['right', 'up', 'down'],
                  'G': ['left', 'down'],
                  'H': ['right', 'left', 'down'],
                  'I': ['right', 'down']}
    # make sure the state is in string form
    str_state = str(state)

    # set the action list according to where the zero is located in the state representation
    if str_state[0] == '0':
        zero_location = 'A'
        action_list = action_dic[zero_location]
    elif str_state[1] == '0':
        zero_location = 'B'
        action_list = action_dic[zero_location]
    elif str_state[2] == '0':
        zero_location = 'C'
        action_list = action_dic[zero_location]
    elif str_state[3] == '0':
        zero_location = 'D'
        action_list = action_dic[zero_location]
    elif str_state[4] == '0':
        zero_location = 'E'
        action_list = action_dic[zero_location]
    elif str_state[5] == '0':
        zero_location = 'F'
        action_list = action_dic[zero_location]
    elif str_state[6] == '0':
        zero_location = 'G'
        action_list = action_dic[zero_location]
    elif str_state[7] == '0':
        zero_location = 'H'
        action_list = action_dic[zero_location]
    elif str_state[8] == '0':
        zero_location = 'I'
        action_list = action_dic[zero_location]
    else:
        action_list = []
        print('cant find zero')

    return action_list


class Node:
    def __init__(self, state, parent, action, path_cost, h_value):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.h_value = h_value


def child_node(node, action):
    # convert the state to str so that it can be indexed.
    state = str(node.state)
    a = int(state[0])
    b = int(state[1])
    c = int(state[2])
    d = int(state[3])
    e = int(state[4])
    f = int(state[5])
    g = int(state[6])
    h = int(state[7])
    i = int(state[8])

    # build an array of the state
    state_array = np.array([[a, b, c], [d, e, f], [g, h, i]])
    # find the index of the zero in the array
    min_index = np.argmin(state_array)
    zero_loc = [min_index // 3, min_index % 3]

    # execute the action on the state array
    if action == 'right':
        state_array[zero_loc[0], zero_loc[1]] = state_array[zero_loc[0], zero_loc[1] - 1]
        state_array[zero_loc[0], zero_loc[1]-1] = 0
    elif action == 'left':
        state_array[zero_loc[0], zero_loc[1]] = state_array[zero_loc[0], zero_loc[1] + 1]
        state_array[zero_loc[0], zero_loc[1]+1] = 0
    elif action == 'up':
        state_array[zero_loc[0], zero_loc[1]] = state_array[zero_loc[0] + 1, zero_loc[1]]
        state_array[zero_loc[0]+1, zero_loc[1]] = 0
    elif action == 'down':
        state_array[zero_loc[0], zero_loc[1]] = state_array[zero_loc[0] - 1, zero_loc[1]]
        state_array[zero_loc[0]-1, zero_loc[1]] = 0
    else:
        print('Unknown action')

    # put the final state back into a string
    final_state = str(state_array[0, 0])+str(state_array[0, 1])+str(state_array[0, 2])+str(state_array[1, 0])\
        + str(state_array[1, 1]) + str(state_array[1, 2])+str(state_array[2, 0])+str(state_array[2, 1])\
        + str(state_array[2, 2])

    # set the path cost of the child to be 1 greater than the parent
    path_cost = node.path_cost + 1

    # construct the child node
    child = Node(final_state, node, action, path_cost, 0)

    return child


def depth_limit_search(node, limit):
    # if the input state is the goal then return the input node
    if node.state == goal:
        print('input is the goal')
        return node

    # initialize the frontier to include the input node
    frontier = [node]
    # initialize the flags for the while loop
    goal_complete = 0

    # while the goal state has not been reached and the frontier is not empty
    while not goal_complete and len(frontier) != 0:
        # extract the last node from the frontier
        leaf = frontier.pop()
        # get the list of available actions from the state of the node
        action_list = get_actions(leaf.state)
        # if the path cost is below the depth limit
        if leaf.path_cost < limit:
            # for each available action
            for i in action_list:
                # generate the child of the action
                child = child_node(leaf, i)
                # if the child reached the goal state then return the child
                if child.state == goal:
                    goal_complete = 1
                    return child
                # else add the child to the frontier if it is not already there
                elif child not in frontier:
                    frontier.append(child)

    return


def iterative_deepening(node, max_depth):

    # initialize flag for the while loop and the depth for the iterative deepening
    result = 0
    depth = 0

    # while no goal node has been found and the max search depth has not been reached
    while not result and depth < max_depth:
        # execute the depth limited search at current depth
        result = depth_limit_search(node, depth)
        # increment the depth
        depth += 1

    # if no goal state was found then return nothing and print out a message
    if not result:
        print('no solution found')
        return
    # else if a goal node has been found
    elif result:
        # select the node that matched the goal node
        node = result
        # find the path cost/number of steps require to reach that node
        path_cost = node.path_cost
        # initialize an empty action list
        action_list = list()
        # while the node has a parent node
        while node.parent:
            # select the parent of the node
            parent = node.parent
            # add the action required to get to the child node to the action list
            action_list.append(node.action)
            # set the node to its parent
            node = parent
        # reverse the action list to get it in the correct order from initial state to goal state.
        action_list.reverse()

    return action_list, path_cost


def num_wrong_tiles(state):
    # initialize the count of number of wrong tiles to zero
    count = 0
    # ensure the state is in string form
    state = str(state)
    # for each digit in the state
    for i in range(len(state)-1):
        # if the digit in the state matches the digit in the goal then increment the count of number of wrong tiles
        if state[i] != goal[i]:
            count += 1
    return count


def manhattan_distance(state):
    # initialize goal and current state dictionaries
    goal_dic = {}
    current_dic = {}
    # initialize the total distance away from the goal for all tiles
    value = 0
    # for each position in the puzzle
    for i in range(len(state)):
        # add the digit of the position as a key with its position as the values of the key
        goal_dic.update({goal[i]: [i // 3, i % 3]})
        current_dic.update({state[i]: [i // 3, i % 3]})
    # for each position in the puzzle
    for x in range(len(state)):
        # select the key to be digit in position x of the current state
        key = str(state[x])
        # get the coordinates for the digit selected in the goal and current states
        goal_x, goal_y = goal_dic[key]
        current_x, current_y = current_dic[key]
        # calculate the x and y distance between the goal state and the current state for the digit x
        dist_x = goal_x - current_x
        dist_y = goal_y - current_y
        # increase the total distance by the x and y distance of digit x
        value += abs(dist_x) + abs(dist_y)

    return value


def astar(node, heuristic):
    # if the input is the goal state then return the no action and 0 path cost
    if node.state == goal:
        print('input is the goal')
        action_list = 'none'
        path_cost = 0
        goal_complete = 1
        return action_list, path_cost
    # initialize the frontier to contain the input node
    frontier = [node]
    # initialize the explored set and explored state to be empty
    explored_set = set()
    explored_states = set()
    # initialize loop flags
    goal_complete = 0

    # while the goal node hasnt been found and the frontier isnt empty
    while not goal_complete and len(frontier) != 0:
        # select the leaf node from the frontier
        minleaf = frontier.pop
        leaf = minleaf(0)
        # get available actions for the leaf node
        action_list = get_actions(leaf.state)
        # for each available action
        for i in action_list:
            # generate the child of the action
            child = child_node(leaf, i)
            # if the child is a goal node then stop the loop and set the child to be the final node
            if child.state == goal:
                goal_complete = 1
                final_node = child
            # if the child is not in the explored set and frontier and the child state hasnt been explored
            elif child not in explored_set and child not in frontier and child.state not in explored_states:
                # if the input heuristic is the number of wrong tiles
                if heuristic == num_wrong_tiles:
                    # set the heuristic value of the child node to be the value of the heuristic plus its path cost
                    child.h_value = num_wrong_tiles(child.state) + child.path_cost
                    # add the child to the frontier
                    frontier.append(child)
                # else if the input heuristic is the manhattan distance
                elif heuristic == manhattan_distance:
                    # set the heuristic value of the child node to be the value of the heuristic plus its path cost
                    child.h_value = manhattan_distance(child.state) + child.path_cost
                    # add the child to the frontier
                    frontier.append(child)
                # add the child state to the explored states
                explored_states.add(child.state)
        # sort the frontier by the value of its heuristic+path cost
        frontier.sort(key=lambda frontier: frontier.h_value)
        # add the leaf to explored nodes
        explored_set.add(leaf)
    # set the output variable to be the path cost of the goal node
    path_cost = final_node.path_cost
    # initialize an empty action list
    action_list = list()
    # while the node has a parent
    while final_node.parent:
        # extract the parent node from the current node
        parent = final_node.parent
        # add the node action to the action list
        action_list.append(final_node.action)
        # set the current node to the parent node
        final_node = parent
    # reverse the action list to get it in the correct order from initial state to goal state.
    action_list.reverse()

    return action_list, path_cost


if __name__ == '__main__':
    goal = '123804765'
    # take the argument from the command line
    initial_state = str(sys.argv[1])
    # make sure the command line argument matches what is expected as input for this puzzle
    # exit if there is more than one input argument
    if len(sys.argv) > 2:
        print('Too many input arguments. Make sure there are no spaces in your input number')
        exit()
    # exit if there are too few input arguments
    elif len(sys.argv) < 1:
        print('Too few input arguments. This program requires and input of an initial state')
        exit()
    # exit if there are too many digits in the input state
    elif len(initial_state) != 9:
        print('Please input a state with the correct number of digits (9)')
        exit()
    # exit if the digit values of the input state dont match the values of the goal state
    elif sorted(goal) != sorted(initial_state):
        print('Please make sure no digits are repeated and there are no "9" digits')
        exit()

    # initialize the first node with the initial state, 0 parent node, no action, 0 path cost, and 0 heuristic value
    first_node = Node(initial_state, 0, '', 0, 0)
    # get the start time of the iterative deepening algorithm
    iter_deep_start = time.time()
    # run the iterative deepening algorithm with a max depth of 50
    iter_deep = iterative_deepening(first_node, 50)
    # get the end time of the iterative deepening algorithm
    iter_deep_end = time.time()

    # get the start time of the a* num wrong tiles algorithm
    a_num_start = time.time()
    # run the a* num wrong tiles algorithm
    a_numwrong = astar(first_node, num_wrong_tiles)
    # get the end time of the a* num wrong tiles algorithm
    a_num_end = time.time()

    # get the start time of the a* manhattan distance algorithm
    a_man_start = time.time()
    # run the a* manhattan distance algorithm
    a_manhattan = astar(first_node, manhattan_distance)
    # get the end time fo the a* manhattan distance algorithm
    a_man_end = time.time()

    # calculate the execution time of each algorithm
    iter_deep_time = iter_deep_end - iter_deep_start
    a_num_time = a_num_end - a_num_start
    a_man_time = a_man_end - a_man_start

    # print the results of each algorithm
    print('\nIterative Deepening Results \nPath Cost: ', iter_deep[1], '\nPath: ', iter_deep[0])
    print('execution time = ', iter_deep_time)
    print('\nA* num_wrong_tiles Results \nPath Cost: ', a_numwrong[1], '\nPath: ', a_numwrong[0])
    print('execution time = ', a_num_time)
    print('\nA* manhattan_distance Results \nPath Cost: ', a_manhattan[1], '\nPath: ', a_manhattan[0])
    print('execution time = ', a_man_time)

