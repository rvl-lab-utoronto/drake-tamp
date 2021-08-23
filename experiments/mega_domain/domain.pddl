(define (domain mega_domain)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates 
        (arm ?arm)
        (item ?item)
        (region ?region)
        (worldpose ?item ?X_WI)
        (handpose ?item ?X_HI)
        (conf ?arm ?q)
        (graspconf ?arm ?q); we dont plan motions with these
        (sink ?region)
        (burner ?region)

        (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q)
        (motion ?arm ?q1 ?traj ?q2)
        (colfree-empty ?arm ?q ?item ?X_WI)
        (colfree-holding ?arm ?q ?item ?X_WI)
        (region-support ?item ?X_WI ?region)
        (item-support ?upperitem ?X_WU ?loweritem ?X_WL)

        ; fluents 
        (empty ?arm)
        (atconf ?arm ?q)
        (atworldpose ?item ?X_WI)
        (athandpose ?arm ?item ?X_HI)
        (clear ?item)
        (on-region ?item ?region)
        (on-item ?upperitem ?loweritem)
        (cooked ?item)
        (clean ?item)

        (item-safe-empty ?arm ?q ?item)
        (item-safe-holding ?arm ?q ?item)
        (grasped ?arm ?item)
    )

    (:derived (grasped ?arm ?item) (exists (?X_HI) (and (item ?item) (arm ?arm) (athandpose  ?arm ?item ?X_HI))))

    (:derived (item-safe-empty ?arm ?q ?item)
        (and
            (item ?item) (arm ?arm) (graspconf ?arm ?q)
        (or
            (exists (?X_HI)
                (and
                    (handpose ?item ?X_HI)
                    (athandpose ?arm ?item ?X_HI)
                )
            ) 
            (exists (?X_WI)
                (and
                    (colfree-empty ?arm ?q ?item ?X_WI)
                    (atworldpose ?item ?X_WI) 
                ) 
            ) 
        )
        )
    )

    (:derived (item-safe-holding ?arm ?q ?item)
        (and
            (item ?item) (arm ?arm) (graspconf ?arm ?q)
        (or
            (exists (?X_HI)
                (and
                    (handpose ?item ?X_HI)
                    (athandpose ?arm ?item ?X_HI)
                )
            ) 
            (exists (?X_WI)
                (and
                    (colfree-holding ?arm ?q ?item ?X_WI)
                    (atworldpose ?item ?X_WI) 
                ) 
            ) 
        )
        )
    )

    (:action pick 
        :parameters (?arm ?item ?region ?X_WI ?X_HI ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (region ?region)
            (clear ?item)
            (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q)
            (atworldpose ?item ?X_WI)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-region ?item ?region)
            (forall (?otherblock)
                (imply 
                    (and (item ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (item-safe-empty ?arm ?q ?otherblock)
                )
            )
        ) 
        :effect (and
            (athandpose ?arm ?item ?X_HI)
            (not (atworldpose ?item ?X_WI))
            (not (empty ?arm))
            (not (on-region ?item ?region))
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

    (:action place
        :parameters (?arm ?item ?region ?X_WI ?X_HI ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (item ?item)
            (region ?region)
            (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q)
            (athandpose ?arm ?item ?X_HI)
            (atconf ?arm ?pre_q)
            (region-support ?item ?X_WI ?region)
            (forall (?otherblock)
                (imply 
                    (and (item ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (item-safe-holding ?arm ?q ?otherblock)
                )
            )
        ) 
        :effect (and
            (not (athandpose ?arm ?item ?X_HI))
            (atworldpose ?item ?X_WI) 
            (empty ?arm)
            (on-region ?item ?region)
        )
    )

    (:action stack
        :parameters (?arm ?item ?X_WI ?X_HI ?loweritem ?X_WL ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (item ?item)
            (item ?loweritem)
            (clear ?loweritem)
            (not (= ?item ?loweritem))
            (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q) 
            (athandpose ?arm ?item ?X_HI)
            (atworldpose ?loweritem ?X_WL)
            (atconf ?arm ?pre_q)
            (item-support ?item ?X_WI ?loweritem ?X_WL)
            (forall (?otherblock)
                (imply 
                    (and (item ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (item-safe-holding ?arm ?q ?otherblock)
                )
            )
        )
        :effect (and
            (empty ?arm)
            (not (clear ?loweritem))
            (not (athandpose ?arm ?item ?X_HI))
            (atworldpose ?item ?X_WI) 
            (on-item ?item ?loweritem)
        )
    )

    (:action unstack
        :parameters (?arm ?item ?X_WI ?X_HI ?loweritem ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (item ?loweritem)
            (clear ?item)
            (not (= ?item ?loweritem))
            (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q)
            (atworldpose ?item ?X_WI)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-item ?item ?loweritem)
            (forall (?otherblock)
                (imply 
                    (and (item ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (item-safe-empty ?arm ?q ?otherblock)
                )
            )
        ) 
        :effect (and
            (athandpose ?arm ?item ?X_HI)
            (clear ?loweritem)
            (not (atworldpose ?item ?X_WI))
            (not (empty ?arm))
            (not (on-item ?item ?loweritem))
        )
    )

    (:action wash
        :parameters (?item ?region) 
        :precondition (and
            (region ?region)
            (item ?item) 
            (sink ?region) 
            (on-region ?item ?region) 
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
            (on-region ?item ?region)
        )
        :effect (and
            (cooked ?item)
        )
    )
    
)