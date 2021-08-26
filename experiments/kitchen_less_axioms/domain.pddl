(define (domain kitchen)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality :negative-preconditions)
    (:predicates 
        ;static predicates
        (item ?item)
        (region ?region)
        (sink ?region)
        (burner ?region)
        (cooked ?item)
        (clean ?item)
        ; I: item frame, W: world frame, H: hand frame
        (worldpose ?item ?X_WI)
        (handpose ?item ?X_HI)
        ;7 DOF conf 
        (conf ?q)
        (graspconf ?q)
        ;sequence of 7 DOF confs, linearly interpolated
        ; if item where at X_WI, would it be in region?
        (contained ?item ?X_WI ?region)
        (motion ?q1 ?traj ?q2)
        (ik ?item ?X_WI ?X_HI ?pre_q ?q)
        ; TODO check collisiosn between placed items and other items
        (colfree ?q ?item ?X_WI)
        ;(colfreeholding ?q ?itemholding ?X_HI ?item ?X_WI)

        ;fluents predicates
        (atconf ?q)
        (atpose ?item ?X_WI)
        (holding ?item ?X_HI)
        (empty)

        (in ?item ?region)

        ;derived
        (safe ?q ?item)
        (grasped ?item)
        ;(safeplace ?q ?itemholding ?X_HI ?item)
    )

    ;(:functions
        ;(distance ?traj)
    ;)

    (:derived (safe ?q ?item) 
        (or
            (exists (?X_HI)
                (and
                    (holding ?item ?X_HI)
                    (handpose ?item ?X_HI)
                )
            ) 
            (exists (?X_WI)
                (and
                    (colfree ?q ?item ?X_WI)
                    (atpose ?item ?X_WI) 
                ) 
            ) 
        )
    )

    (:derived (grasped ?item) (exists (?X_HI) (holding ?item ?X_HI)))

    (:action move
        :parameters(?q1 ?traj ?q2) 
        :precondition (and 
            (motion ?q1 ?traj ?q2)
            (conf ?q1)
            (conf ?q2)
            (atconf ?q1)
            (or
                (empty)
                (forall (?item)
                    (or
                        (grasped ?item)
                        (not (exists
                                (?X_WI ?X_HI ?q)
                                (and
                                    (ik ?item ?X_WI ?X_HI ?q2 ?q)
                                    (atpose ?item ?X_WI)
                                )    
                            )
                        )
                    )
                )
            )
        )
        :effect (and 
            (atconf ?q2)
            (not (atconf ?q1))
            ;(increase (total-cost) (distance ?traj)) 
            ; TODO(agro): add cost here
        )
    )

    (:action pick
        :parameters (?item ?region ?X_WI ?X_HI ?pre_q ?q)
        :precondition (and 
            (ik ?item ?X_WI ?X_HI ?pre_q ?q)
            (region ?region)
            (item ?item)
            (worldpose ?item ?X_WI)
            (handpose ?item ?X_HI)
            (atpose ?item ?X_WI)
            (graspconf ?q)
            (conf ?pre_q)
            (in ?item ?region)
            (empty)
            (atconf ?pre_q)
            (forall (?otheritem)
                (imply 
                    (item ?otheritem) 
                    (safe ?q ?otheritem)
                ) 
            )
        )
        :effect (and
            (holding ?item ?X_HI)
            (not (atpose ?item ?X_WI))
            (not (empty))
            (not (in ?item ?region))
        )
    )

    (:action place
        :parameters (?item ?region ?X_WI ?X_HI ?pre_q ?q)
        :precondition (and 
            (ik ?item ?X_WI ?X_HI ?pre_q ?q)
            (holding ?item ?X_HI)
            (conf ?pre_q)
            (graspconf ?q)
            (atconf ?pre_q)
            (item ?item)
            (region ?region)
            (contained ?item ?X_WI ?region)
            (worldpose ?item ?X_WI)
            (handpose ?item ?X_HI)
            (forall (?otheritem)
                (imply 
                    (item ?otheritem) 
                    (safe ?q ?otheritem)
                ) 
            )
        )
        :effect (and
            (not (holding ?item ?X_HI))
            (atpose ?item ?X_WI) 
            (empty)
            (in ?item ?region)
        )    
    )

    (:action wash
        :parameters (?item ?region) 
        :precondition (and
            (region ?region)
            (item ?item) 
            (sink ?region) 
            (in ?item ?region) 
        )
        :effect (and
            (clean ?item)
        )
    )

    (:action cook
        :parameters (?item ?region) 
        :precondition (and 
            (item ?item)
            (region ?region)
            (burner ?region)
            (clean ?item)
            (in ?item ?region)
        )
        :effect (and
            (cooked ?item)
        )
    )


)