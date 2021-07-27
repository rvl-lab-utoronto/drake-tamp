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
        (armsafeconf ?arm ?q) ;a guarenteed no-collision configuration

        ; stream certified
        ; find-ik
        (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
        ; find-motion
        (motion ?arm ?q1 ?traj ?q2)
        ; check-colfree-block
        ; if arm is at q and item at X_WI, are there collisions 
        (colfree-block ?arm ?q ?block ?X_WB)
        ; check-colfree-arms
        ; if arm1 is at q1, and arm2 is at q2, are there collisions?
        (colfree-arms ?arm1 ?q1 ?arm2 ?q2)
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

        ;derived
        (block-safe ?arm ?q ?block)
        (on-block ?upperblock ?lowerblock)
        (on-table ?block ?table)
        (on-any-table ?block ?X_WB)
        (armsafe ?arm ?otherarm)
    )

    (:derived (on-table ?block ?table) 
        (exists (?X_WB) (and
                (table-support ?block ?X_WB ?table) 
                (atworldpose ?block ?X_WB)
            )
        )
    )

    (:derived (armsafe ?arm ?otherarm)
        (or
            (= ?arm ?otherarm) 
            (exists (?q) (and
                    (armsafeconf ?otherarm ?q) 
                    (atconf ?otherarm ?q)
                )
            )
        )
    )

    ; if block was at X_WB, would it be on any table
    (:derived (on-any-table ?block ?X_WB) 
        (exists (?table) (and
                (table-support ?block ?X_WB ?table)
            )
        )
    )

    (:derived (block-safe ?arm ?q ?block) 
        (or
            (exists (?X_HB)
                (and
                    (handpose ?block ?X_HB)
                    (athandpose ?arm ?block ?X_HB)
                )
            ) 
            (exists (?X_WB)
                (and
                    (colfree-block ?arm ?q ?block ?X_WB)
                    (atworldpose ?block ?X_WB) 
                ) 
            ) 
        )
    )

    (:derived (on-block ?upperblock ?lowerblock)
        (exists (?X_WL ?X_WU)
            (and
                (block-support ?upperblock ?X_WU ?lowerblock ?X_WL)
                (atworldpose ?lowerblock ?X_WL)
                (atworldpose ?upperblock ?X_WU)
            )
        )
    )

    (:action pick  ;off of a table
        :parameters (?arm ?block ?X_WB ?X_HB ?pre_q ?q)
        :precondition (and
            (clear ?block)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?block ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-any-table ?block ?X_WB)
            (forall (?otherblock)
                (imply 
                    (block ?otherblock) 
                    (block-safe ?arm ?q ?otherblock)
                ) 
            )
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (not (atworldpose ?block ?X_WB))
            (not (empty ?arm))
        )
    )

    (:action move
        :parameters (?arm ?q1 ?traj ?q2) 
        :precondition (and
            (motion ?arm ?q1 ?traj ?q2) 
            (atconf ?arm ?q1)
            (forall (?otherarm)
                (imply
                    (arm ?otherarm) 
                    (armsafe ?arm ?otherarm)
                ) 
            )
        )
        :effect (and
            (atconf ?arm ?q2)
            (not (atconf ?arm ?q1))
        )
    )

    (:action place ; place block on table
        :parameters (?arm ?block ?X_WB ?X_HB ?pre_q ?q) 
        :precondition (and
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (athandpose ?arm ?block ?X_HB)
            (atconf ?arm ?pre_q)
            (on-any-table ?block ?X_WB)
            (forall (?otherblock)
                (imply 
                    (block ?otherblock) 
                    (block-safe ?arm ?q ?otherblock)
                ) 
            )
        ) 
        :effect (and
            (not (athandpose ?arm ?block ?X_HB))
            (atworldpose ?block ?X_WB) 
            (empty ?arm)
        )
    )

    (:action stack ;place block on lowerblock
        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?X_WL ?pre_q ?q) 
        :precondition (and
            (clear ?lowerblock)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q) 
            (athandpose ?arm ?block ?X_HB)
            (atworldpose ?lowerblock ?X_WL)
            (atconf ?arm ?pre_q)
            (block-support ?block ?X_WB ?lowerblock ?X_WL)
            (forall (?otherblock)
                (imply 
                    (block ?otherblock) 
                    (block-safe ?arm ?q ?otherblock)
                ) 
            )
            ;(forall (?otherarm)
            ;    (imply 
            ;        (arm ?otherarm) 
            ;        (arm-safe ?arm ?pre_q ?otherarm)
            ;    ) 
            ;)
        )
        :effect (and
            (not (clear ?lowerblock))
            (not (athandpose ?arm ?block ?X_HB))
            (atworldpose ?block ?X_WB) 
            (empty ?arm)
        )
    )

    (:action unstack
        :parameters (?arm ?block ?X_WB ?X_HB ?lowerblock ?pre_q ?q)
        :precondition (and
            (clear ?block)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?block ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-block ?block ?lowerblock)
            (forall (?otherblock)
                (imply 
                    (block ?otherblock) 
                    (block-safe ?arm ?q ?otherblock)
                ) 
            )
            ;(forall (?otherarm)
            ;   (imply 
            ;       (arm ?otherarm) 
            ;       (arm-safe ?arm ?pre_q ?otherarm)
            ;   ) 
            ;) 
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (not (atworldpose ?block ?X_WB))
            (not (empty ?arm))
            (clear ?lowerblock)
        )
    )
    
)