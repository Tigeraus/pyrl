#coding: utf-8
"""
[1]	T. Yang and M. N. Shadlen,
“Probabilistic reasoning by neurons,” Nature, vol. 447, no. 7148, pp. 1075–1080, Jun. 2007.

task condition:
model file need modification, gpu needed
/Users/huzi/Workspace/Work2018/pyrl/examples/models/romo train --dt 100.0 --suffix test --dt-save 100.0 --seed 100

"""
from __future__ import division

import numpy as np

from pyrl import tasktools
import copy

# Inputs
inputs = tasktools.to_map('FIXATION', 'TARGET-L', 'TARGET-R', "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")

weights = [-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9]

# Actions
actions = tasktools.to_map('FIXATE', 'L', 'R')


# Trial conditions
targets = ['Red_L', 'Red_R']
sn = 8
def getShapepair():
    shapepairs = []
    for tg in range(2):
        for i in range(sn):
            for j in range(sn):
                for k in range(sn):
                    for l in range(sn):
                        shapepairs.append((tg, i, j, k, l))
    return copy.deepcopy(shapepairs)


shapepairs = getShapepair()

n_conditions = len(targets) * len(shapepairs)

# Training
n_gradient   = n_conditions
n_validation = 20*n_conditions

# Slow down the learning
lr          = 0.002
baseline_lr = 0.002

# Input noise
sigma = np.sqrt(2*100*0.001)


fixation  = 200
shape0 = 500
shape1 = 500
shape2 = 500
shape3 = 500
decision = 500
tmax = fixation + shape0 + shape1 + shape2 + shape3 + decision


def get_condition(rng, dt, context={}):
    #-------------------------------------------------------------------------------------
    # Epochs
    #-------------------------------------------------------------------------------------
    durations = {
        'fixation':     (0, fixation),
        'shape0':       (fixation, fixation + shape0),
        'shape1':       (fixation + shape0, fixation + shape0 + shape1),
        'shape2':       (fixation + shape0 + shape1, fixation + shape0 + shape1 + shape2),
        'shape3':       (fixation + shape0 + shape1 + shape2, fixation + shape0 + shape1 + shape2 + shape3),
        'decision':     (fixation + shape0 + shape1 + shape2 + shape3, tmax),
        'tmax':       tmax
        }

    time, epochs = tasktools.get_epochs_idx(dt, durations)

    target = context.get('target')
    if target is None:
        target = tasktools.choice(rng, targets)

    shapepair = context.get('shapepair')
    if shapepair is None:
        shapepair = tasktools.choice(rng, getShapepair())

    return {
        'durations': durations,
        'time':      time,
        'epochs':    epochs,
        'target':     target,
        'shapepair':     shapepair
        }

# Rewards
R_ABORTED = -1
R_CORRECT = +1


def get_step(rng, dt, trial, t, a):
    #-------------------------------------------------------------------------------------
    # Reward
    #-------------------------------------------------------------------------------------
    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0
    if t-1 not in epochs['decision']:
        if a != actions['FIXATE']:
            status['continue'] = False
            status['choice']   = None
            reward = R_ABORTED
    elif t-1 in epochs['decision']:
        sum_weight = sum(map(lambda x: weights[x], trial['shapepairs']))
        r_l = None
        if trial['targets'] == 'Red_L':
            r_l = 1
            pass
        elif trial['targets'] == 'Red_R':
            r_l = 0
            pass
        else:
            pass
        if sum_weight == 0:
            status['continue'] = False
            if a == actions['L']:
                status['choice'] = 'L'
            elif a == actions['R']:
                status['choice'] = 'R'
            status['correct'] = np.random.choice([0, 1])
            if status['correct']:
                reward = R_CORRECT
            pass
        else:
            if a == actions['L']:
                status['continue'] = False
                status['choice']   = 'L'
                if sum_weight > 0 and r_l:
                    status['correct']  = 1
                else:
                    status['correct'] = 0
                if status['correct']:
                    reward = R_CORRECT
            elif a == actions['R']:
                status['continue'] = False
                status['choice']   = 'R'

                if sum_weight > 0 and (not r_l):
                    status['correct']  = 1
                else:
                    status['correct'] = 0
                if status['correct']:
                    reward = R_CORRECT

    #-------------------------------------------------------------------------------------
    # Inputs
    #-------------------------------------------------------------------------------------

    u = np.zeros(len(inputs))
    # fixation
    if t not in epochs['decision']:
        u[inputs['FIXATION']] = 1

    # target
    if (t in epochs['shape0']) or (t in epochs['shape1']) or (t in epochs['shape2']) or (t in epochs['shape3']):
        if trial['target'] == 'Red_L':
            u[inputs['TARGET-L']] = 1
            u[inputs['TARGET-R']] = 0
        elif trial['target'] == 'Red_R':
            u[inputs['TARGET-L']] = 0
            u[inputs['TARGET-R']] = 1

    # shape
    shape_seq = trial['shapepair']
    if t in epochs['shape0']:
        shape_name = "S"+str(shape_seq[0])
        u[inputs[shape_name]] = 1
    if t in epochs['shape1']:
        shape_name = "S"+str(shape_seq[1])
        u[inputs[shape_name]] = 1
    if t in epochs['shape2']:
        shape_name = "S"+str(shape_seq[2])
        u[inputs[shape_name]] = 1
    if t in epochs['shape3']:
        shape_name = "S"+str(shape_seq[3])
        u[inputs[shape_name]] = 1

    #-------------------------------------------------------------------------------------

    return u, reward, status



def terminate(perf):
    p_decision, p_correct = tasktools.correct_2AFC(perf)

    return p_decision >= 0.99 and p_correct >= 0.97
