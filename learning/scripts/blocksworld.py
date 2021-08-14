
#%%
from learning.gnn.data import query_data, get_pddl_key
import numpy as np
import json
# %%
# blocks_world = get_pddl_key('blocks_world')
blocks_world = "(define (domain blocks_world)\n    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)\n\n    (:predicates \n        ; type/static predicates\n        (arm ?arm)\n        (block ?block)\n        (table ?table)\n        ; find-table-place and find-block-place\n        (worldpose ?block ?X_WB)\n        ; find-grasp\n        (handpose ?block ?X_HB)\n        (conf ?arm ?q)\n        (graspconf ?arm ?q); we dont plan motions with these\n\n        ; stream certified\n        ; find-ik\n        (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)\n        ; find-motion\n        (motion ?arm ?q1 ?traj ?q2)\n        ; check-colfree-block\n        ; if arm is at q and item at X_WI, are there collisions \n        (colfree-block ?arm ?q ?block ?X_WB)\n        ; check-colfree-arms\n        ; if arm1 is at q1, and arm2 is at q2, are there collisions?\n        (colfree-arms ?arm1 ?q1 ?arm2 ?q2)\n        ; find-table-place\n        (table-support ?block ?X_WB ?table)\n        ; find-block-place\n        (block-support ?upperblock ?X_WU ?lowerblock ?X_WL)\n\n        ; fluents \n        (empty ?arm)\n        (atconf ?arm ?q)\n        (atworldpose ?block ?X_WB)\n        (athandpose ?arm ?block ?X_HB)\n        (clear ?block)\n\n        ;derived\n        (block-safe ?arm ?q ?block)\n        (on-block ?upperblock ?lowerblock)\n        (on-table ?block ?table)\n        (on-any-table ?block ?X_WB)\n    )\n\n    (:derived (on-table ?block ?table) \n        (exists (?X_WB) (and\n                (table-support ?block ?X_WB ?table) \n                (atworldpose ?block ?X_WB)\n            )\n        )\n    )\n\n    ; if block was at X_WB, would it be on any table\n    (:derived (on-any-table ?block ?X_WB) \n        (exists (?table) (and\n                (table-support ?block ?X_WB ?table)\n            )\n        )\n    )\n\n    (:derived (block-safe ?arm ?q ?block) \n        (or\n            (exists (?X_HB)\n                (and\n                    (handpose ?block ?X_HB)\n                    (athandpose ?arm ?block ?X_HB)\n                )\n            ) \n            (exists (?X_WB)\n                (and\n                    (colfree-block ?arm ?q ?block ?X_WB)\n                    (atworldpose ?block ?X_WB) \n                ) \n            ) \n        )\n    )\n\n    (:derived (on-block ?upperblock ?lowerblock)\n        (exists (?X_WL ?X_WU)\n            (and\n                (block-support ?upperblock ?X_WU ?lowerblock ?X_WL)\n                (atworldpose ?lowerblock ?X_WL)\n                (atworldpose ?upperblock ?X_WU)\n            )\n        )\n    )\n\n    (:action pick  ;off of a table\n        :parameters (?arm ?block ?X_WB ?X_HB ?pre_q ?q)\n        :precondition (and\n            (clear ?block)\n            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)\n            (atworldpose ?block ?X_WB)\n            (empty ?arm)\n            (atconf ?arm ?pre_q)\n            (on-any-table ?block ?X_WB)\n            (forall (?otherblock)\n                (imply \n                    (block ?otherblock) \n                    (block-safe ?arm ?q ?otherblock)\n                ) \n            )\n        ) \n        :effect (and\n            (athandpose ?arm ?block ?X_HB)\n            (not (atworldpose ?block ?X_WB))\n            (not (empty ?arm))\n        )\n    )\n\n    (:action move\n        :parameters (?arm ?q1 ?traj ?q2) \n        :precondition (and\n            (motion ?arm ?q1 ?traj ?q2) \n            (atconf ?arm ?q1)\n        )\n        :effect (and\n            (atconf ?arm ?q2)\n            (not (atconf ?arm ?q1))\n        )\n    )\n\n    (:action place ; place block on table\n        :parameters (?arm ?block ?X_WB ?X_HB ?pre_q ?q) \n        :precondition (and\n            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)\n            (athandpose ?arm ?block ?X_HB)\n            (atconf ?arm ?pre_q)\n            (on-any-table ?block ?X_WB)\n            (forall (?otherblock)\n                (imply \n                    (block ?otherblock) \n                    (block-safe ?arm ?q ?otherblock)\n                ) \n            )\n        ) \n        :effect (and\n            (not (athandpose ?arm ?block ?X_HB))\n            (atworldpose ?block ?X_WB) \n            (empty ?arm)\n        )\n    )\n\n    (:action stack ;place block on lowerblock\n        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?X_WL ?pre_q ?q) \n        :precondition (and\n            (clear ?lowerblock)\n            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q) \n            (athandpose ?arm ?block ?X_HB)\n            (atworldpose ?lowerblock ?X_WL)\n            (atconf ?arm ?pre_q)\n            (block-support ?block ?X_WB ?lowerblock ?X_WL)\n            (forall (?otherblock)\n                (imply \n                    (block ?otherblock) \n                    (block-safe ?arm ?q ?otherblock)\n                ) \n            )\n            ;(forall (?otherarm)\n            ;    (imply \n            ;        (arm ?otherarm) \n            ;        (arm-safe ?arm ?pre_q ?otherarm)\n            ;    ) \n            ;)\n        )\n        :effect (and\n            (not (clear ?lowerblock))\n            (not (athandpose ?arm ?block ?X_HB))\n            (atworldpose ?block ?X_WB) \n            (empty ?arm)\n        )\n    )\n\n    (:action unstack\n        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?pre_q ?q)\n        :precondition (and\n            (clear ?block)\n            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)\n            (atworldpose ?block ?X_WB)\n            (empty ?arm)\n            (atconf ?arm ?pre_q)\n            (on-block ?block ?lowerblock)\n            (forall (?otherblock)\n                (imply \n                    (block ?otherblock) \n                    (block-safe ?arm ?q ?otherblock)\n                ) \n            )\n            ;(forall (?otherarm)\n            ;   (imply \n            ;       (arm ?otherarm) \n            ;       (arm-safe ?arm ?pre_q ?otherarm)\n            ;   ) \n            ;) \n        ) \n        :effect (and\n            (athandpose ?arm ?block ?X_HB)\n            (not (atworldpose ?block ?X_WB))\n            (not (empty ?arm))\n            (clear ?lowerblock)\n        )\n    )\n    \n)(define (stream blocks_world)\n\n    (:stream find-traj\n        :inputs (?arm ?q1 ?q2) \n        :fluents (atworldpose athandpose atconf)\n        :domain (and\n            (arm ?arm)\n            (conf ?arm ?q1)  \n            (conf ?arm ?q2)  \n        )\n        :outputs (?traj)\n        :certified (and\n            (motion ?arm ?q1 ?traj ?q2) \n        )\n    )\n    \n    (:stream find-grasp\n        :inputs (?block)\n        :domain (and\n            (block ?block) \n        ) \n        :outputs (?X_HB)\n        :certified (and\n            (handpose ?block ?X_HB)\n        )\n    )\n\n    (:stream find-ik\n        :inputs (?arm ?block ?X_WB ?X_HB)\n        :domain (and\n            (arm ?arm)\n            (worldpose ?block ?X_WB) \n            (handpose ?block ?X_HB)\n        ) \n        :outputs (?pre_q ?q)\n        :certified (and\n            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q) \n            (conf ?arm ?pre_q)\n            (graspconf ?arm ?q)\n        )\n    )\n\n    (:stream check-colfree-block\n        :inputs (?arm ?q ?block ?X_WB)\n        :domain (and\n            (arm ?arm)\n            (block ?block) \n            (graspconf ?arm ?q)\n            (worldpose ?block ?X_WB)\n        )\n        :certified (and\n            (colfree-block ?arm ?q ?block ?X_WB)    \n        )\n    )\n\n    (:stream find-table-place\n        :inputs (?block ?table) \n        :domain (and\n            (block ?block)\n            (table ?table) \n        )\n        :outputs (?X_WB)\n        :certified (and\n            (worldpose ?block ?X_WB)    \n            (table-support ?block ?X_WB ?table)\n        )\n    )\n\n    (:stream find-block-place\n        :inputs (?block ?lowerblock ?X_WL) \n        :domain (and\n            (block ?block) \n            (block ?lowerblock) \n            (worldpose ?lowerblock ?X_WL)\n        )\n        :outputs (?X_WB)\n        :certified (and\n            (worldpose ?block ?X_WB) \n            (block-support ?block ?X_WB ?lowerblock ?X_WL)\n        )\n    )\n\n)"
# %%
data = query_data(blocks_world, [])
print(len(data))

# %%
num_valid = min(15, len(data) // 5)
np.random.shuffle(data)
valid = data[-num_valid:]
train = data[:-num_valid]
# %%
with open('learning/data/experiments/blocksworld_complexitycollector_lazy.json', 'w') as f:
    json.dump(dict(
        train=train,
        validation=valid,
        experiment_description="Blocks world experiment with data coming from complexitycollector lazy mode."
    ), f, indent=4, sort_keys=True)
