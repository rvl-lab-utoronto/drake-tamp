(define (domain blocks_world)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates 
        ; type/static predicates
        (arm ?arm)
        ;(table ?table) ; we do not care about specific tables in this world
        (block ?block)
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
        ; check-colfree-block
        ; if arm is at q and item at X_WI, are there collisions 
        (colfree-block ?arm ?q ?block ?X_WB)
        ; check-colfree-arms
        ; if arm1 is at q1, and arm2 is at q2, are there collisions?
        (colfree-arms ?arm1 ?q1 ?arm2 ?q2)
        ; find-table-place
        (table-support ?block ?X_WB)
        ; find-block-place
        (block-support ?lowerblock ?X_WL ?upperblock ?X_WU)

        ; fluents 
        (empty ?arm)
        (atconf ?arm ?q)
        (atworldpose ?block ?X_WB)
        (athandpose ?arm ?block ?X_HB)
        (clear ?block)

        ;derived
        (block-safe ?arm ?q ?block)
        (on-block ?lowerblock ?upperblock)
        ;(arm-safe ?arm1 ?q1 ?arm2)
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

    ;(:derived (arm-safe ?arm ?q ?otherarm) 
    ;    (exists (?q_other)
    ;        (and
    ;            (atconf ?otherarm ?q_other) 
    ;            (colfree-arms ?arm ?q ?otherarm ?q_other)
    ;        ) 
    ;    ) 
    ;)

    (:derived (on-block ?lowerblock ?upperblock)
        (exists (?X_WL ?X_WU)
            (and
                (block-support ?lowerblock ?X_WL ?upperblock ?X_WU) 
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
            (table-support ?block ?X_WB)
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
            (table-support ?block ?X_WB)
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
            (block-support ?lowerblock ?X_WL ?block ?X_WB)
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
            (on-block ?lowerblock ?block)
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