(define (domain blocks_world)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates 
        ; type/static predicates
        (arm ?arm)
        (block ?block)
        (table ?table)

        ; fluents 
        (empty ?arm)
        (clear ?block)
        (on-table ?block ?table)
        (on-block ?block ?block)
        (grasped ?arm ?block)
    )

    (:action pick  ;off of a table
        :parameters (?arm ?block ?table)
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (clear ?block)
            (empty ?arm)
            (on-table ?block ?table)
        ) 
        :effect (and
            (grasped ?arm ?block)
            (not (empty ?arm))
            (not (on-table ?block ?table))
        )
    )

    (:action place ; place block on table
        :parameters (?arm ?block ?table) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (table ?table)
            (grasped ?arm ?block)
        ) 
        :effect (and
            (empty ?arm)
            (on-table ?block ?table)
            (not (grasped ?arm ?block))
        )
    )

    (:action stack ;place block on lowerblock
        :parameters (?arm ?block ?lowerblock) 
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (clear ?lowerblock)
            (grasped ?arm ?block)
        )
        :effect (and
            (empty ?arm)
            (not (clear ?lowerblock))
            (not (grasped ?arm ?block))
            (on-block ?block ?lowerblock)
        )
    )

    (:action unstack
        :parameters (?arm ?block ?lowerblock)
        :precondition (and
            (arm ?arm)
            (block ?block)
            (block ?lowerblock)
            (clear ?block)
            (empty ?arm)
            (on-block ?block ?lowerblock)
        ) 
        :effect (and
            (grasped ?arm ?block)
            (clear ?lowerblock)
            (not (empty ?arm))
            (not (on-block ?block ?lowerblock))
        )
    )
    
)