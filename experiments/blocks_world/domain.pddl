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
        ; (atconf ?arm ?q)
        (atworldpose ?block ?X_WB)
        (athandpose ?arm ?block ?X_HB)
        (clear ?block)
        (on-table ?block ?table)
        (on-block ?block ?block)

        ;derived
        (block-safe ?arm ?q ?block)
        (grasped ?arm ?block)
    )
    (:derived (grasped ?arm ?block) (exists (?X_HB) (and (block ?block) (arm ?arm) (athandpose  ?arm ?block ?X_HB))))

    (:derived (block-safe ?arm ?q ?block)
        (and
            (block ?block) (arm ?arm) (graspconf ?arm ?q)
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
            ; (atconf ?arm ?pre_q)
            (on-table ?block ?table)
            (forall (?otherblock)
                (imply 
                    (and (block ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (block-safe ?arm ?q ?otherblock)
                )
            )
        ) 
        :effect (and
            (athandpose ?arm ?block ?X_HB)
            (not (atworldpose ?block ?X_WB))
            (not (empty ?arm))
            (not (on-table ?block ?table))
        )
    )

    ; (:action move
    ;     :parameters (?arm ?q1 ?traj ?q2) 
    ;     :precondition (and
    ;         (motion ?arm ?q1 ?traj ?q2) 
    ;         (atconf ?arm ?q1)
    ;         (or
    ;             (empty ?arm)
    ;             (forall (?item)
    ;                 (or
    ;                     (grasped ?arm ?item)
    ;                     (not (exists
    ;                             (?X_WI ?X_HI ?q)
    ;                             (and
    ;                                 (ik ?arm ?item ?X_WI ?X_HI ?q2 ?q)
    ;                                 (atworldpose ?item ?X_WI)
    ;                             )    
    ;                         )
    ;                     )
    ;                 )
    ;             )
    ;         )
    ;     )
    ;     :effect (and
    ;         (atconf ?arm ?q2)
    ;         (not (atconf ?arm ?q1))
    ;     )
    ; )

    (:action place ; place block on table
        :parameters (?arm ?block ?table ?X_WB ?X_HB ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q)
            (athandpose ?arm ?block ?X_HB)
            ; (atconf ?arm ?pre_q)
            (table-support ?block ?X_WB ?table)
            (forall (?otherblock)
                (imply 
                    (and (block ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (block-safe ?arm ?q ?otherblock)
                )
            )
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
            ; (atconf ?arm ?pre_q)
            (block-support ?block ?X_WB ?lowerblock ?X_WL)
            (forall (?otherblock)
                (imply 
                    (and (block ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (block-safe ?arm ?q ?otherblock)
                )
            )
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
            ; (atconf ?arm ?pre_q)
            (on-block ?block ?lowerblock)
            (forall (?otherblock)
                (imply 
                    (and (block ?otherblock) (arm ?arm) (graspconf ?arm ?q))
                    (block-safe ?arm ?q ?otherblock)
                )
            )
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