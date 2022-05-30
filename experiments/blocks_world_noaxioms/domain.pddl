(define (domain blocks_world)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates 
        ; type/static predicates
        (arm ?arm)
        (block ?block)
        (table ?table)
        ; find-table-place and find-block-place
        (worldpose ?block ?X_WB)
        ; find-grasp
        (handpose ?block ?X_HB)
        (conf ?arm ?q)
        (graspconf ?arm ?q); we dont plan motions with these

        ; stream certified
        ; find-ik
        (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
        ; find-motion
        (motion ?arm ?q1 ?traj ?q2)

        (safe-free ?arm ?q)
        (safe-holding ?arm ?q ?block ?X_HB)

        ; find-table-place
        (table-support ?block ?X_WB ?table)
        ; find-block-place
        (block-support ?upperblock ?X_WU ?lowerblock ?X_WL)

        ; fluents 
        (empty ?arm)
        (atconf ?arm ?q)
        (atworldpose ?block ?X_WB)
        (athandpose ?arm ?block ?X_HB)
        (clear ?block)
        (on-table ?block ?table)
        (on-block ?block ?block)
    )

    (:action pick  ;off of a table
        :parameters (?arm ?block ?table ?X_WB ?X_HB ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (clear ?block)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?block ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-table ?block ?table)
            (safe-free ?arm ?q)
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (not (atworldpose ?block ?X_WB))
            (not (empty ?arm))
            (not (on-table ?block ?table))
        )
    )

    (:action move
        :parameters (?arm ?q1 ?traj ?q2) 
        :precondition (and
            (motion ?arm ?q1 ?traj ?q2) 
            (atconf ?arm ?q1)

        )
        :effect (and
            (atconf ?arm ?q2)
            (not (atconf ?arm ?q1))
        )
    )

    (:action place ; place block on table
        :parameters (?arm ?block ?table ?X_WB ?X_HB ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (athandpose ?arm ?block ?X_HB)
            (atconf ?arm ?pre_q)
            (table-support ?block ?X_WB ?table)
            (safe-holding ?arm ?q ?block ?X_HB)
        ) 
        :effect (and
            (not (athandpose ?arm ?block ?X_HB))
            (atworldpose ?block ?X_WB) 
            (empty ?arm)
            (on-table ?block ?table)
        )
    )

    (:action stack ;place block on lowerblock
        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?X_WL ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (clear ?lowerblock)
            (not (= ?block ?lowerblock))
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q) 
            (athandpose ?arm ?block ?X_HB)
            (atworldpose ?lowerblock ?X_WL)
            (atconf ?arm ?pre_q)
            (block-support ?block ?X_WB ?lowerblock ?X_WL)
            (safe-holding ?arm ?q ?block ?X_HB)
        )
        :effect (and
            (empty ?arm)
            (not (clear ?lowerblock))
            (not (athandpose ?arm ?block ?X_HB))
            (atworldpose ?block ?X_WB) 
            (on-block ?block ?lowerblock)
        )
    )

    (:action unstack
        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (clear ?block)
            (not (= ?block ?lowerblock))
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?block ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-block ?block ?lowerblock)
            (safe-free ?arm ?q)
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (clear ?lowerblock)
            (not (atworldpose ?block ?X_WB))
            (not (empty ?arm))
            (not (on-block ?block ?lowerblock))
        )
    )
    
)