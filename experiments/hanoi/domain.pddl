(define (domain hanoi)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates 
        ; type/static predicates
        (arm ?arm)
        (disc ?disc)
        (peg ?peg)
        ; find-peg-place and find-disc-place
        (worldpose ?disc ?X_WB)
        ; find-grasp
        (handpose ?disc ?X_HB)
        (conf ?arm ?q)
        (graspconf ?arm ?q); we dont plan motions with these
        (smaller ?smallerdisc ?largerdisc)
        (empty-peg ?peg)

        (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q)
        (motion ?arm ?q1 ?traj ?q2)
        ; if arm is at q and item at X_WI, are there collisions 
        ;(colfree-disc ?arm ?q ?disc ?X_WB) ; lets assume this is given for hanoi (For now)
        (peg-support ?disc ?X_WB ?peg)
        ; find-disc-place
        (disc-support ?upperdisc ?X_WU ?lowerdisc ?X_WL)

        ; fluents 
        (empty ?arm)
        (atconf ?arm ?q)
        (atworldpose ?disc ?X_WB)
        (athandpose ?arm ?disc ?X_HB)
        (clear ?disc)
        (on-peg ?disc ?peg)
        (on-disc ?upperdisc ?lowerdisc)

        ;derived
        ;(disc-safe ?arm ?q ?disc)
        (grasped ?arm ?disc)
    )

    (:derived (grasped ?arm ?disc) (exists (?X_HB) (and (disc ?disc) (arm ?arm) (athandpose  ?arm ?disc ?X_HB))))

    ;(:derived (disc-safe ?arm ?q ?disc)
        ;(and
            ;(disc ?disc) (arm ?arm) (graspconf ?arm ?q)
        ;(or
            ;(exists (?X_HB)
                ;(and
                    ;(handpose ?disc ?X_HB)
                    ;(athandpose ?arm ?disc ?X_HB)
                ;)
            ;) 
            ;(exists (?X_WB)
                ;(and
                    ;(colfree-disc ?arm ?q ?disc ?X_WB)
                    ;(atworldpose ?disc ?X_WB) 
                ;) 
            ;) 
        ;)
        ;)
    ;)

    (:action pick  ;off of apeg 
        :parameters (?arm ?disc ?peg ?X_WB ?X_HB ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (disc ?disc)
            (peg ?peg)
            (clear ?disc)
            (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?disc ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
            (on-peg ?disc ?peg)
            (worldpose ?disc ?X_WB)
            (handpose ?disc ?X_HB)
        ) 
        :effect (and
            (athandpose ?arm ?disc ?X_HB)
            (not (atworldpose ?disc ?X_WB))
            (not (empty ?arm))
            (empty-peg ?peg)
            (not (on-peg ?disc ?peg))
        )
    )

    (:action move
        :parameters (?arm ?q1 ?traj ?q2) 
        :precondition (and
            (conf ?arm ?q1)
            (conf ?arm ?q2)
            (motion ?arm ?q1 ?traj ?q2) 
            (atconf ?arm ?q1)
        )
        :effect (and
            (atconf ?arm ?q2)
            (not (atconf ?arm ?q1))
        )
    )

    (:action place ; place disc onpeg 
        :parameters (?arm ?disc ?peg ?X_WB ?X_HB ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (disc ?disc)
            (peg ?peg)
            (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q)
            (athandpose ?arm ?disc ?X_HB)
            (atconf ?arm ?pre_q)
            (peg-support ?disc ?X_WB ?peg)
            (empty-peg ?peg)
            (worldpose ?disc ?X_WB)
            (handpose ?disc ?X_HB)
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        ) 
        :effect (and
            (not (athandpose ?arm ?disc ?X_HB))
            (atworldpose ?disc ?X_WB) 
            (empty ?arm)
            (on-peg ?disc ?peg)
            (not (empty-peg ?peg))
        )
    )

    (:action stack ;place disc onlowerdisc 
        :parameters (?arm ?disc ?X_WB ?X_HB ?lowerdisc ?X_WL ?pre_q ?q) 
        :precondition (and
            (arm ?arm)
            (disc ?disc)
            (disc ?lowerdisc)
            (clear ?lowerdisc)
            (not (= ?disc ?lowerdisc))
            (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q) 
            (athandpose ?arm ?disc ?X_HB)
            (atworldpose ?lowerdisc ?X_WL)
            (atconf ?arm ?pre_q)
            (disc-support ?disc ?X_WB ?lowerdisc ?X_WL)
            (smaller ?disc ?lowerdisc)
            (worldpose ?disc ?X_WB)
            (worldpose ?lowerdisc ?X_WL)
            (handpose ?disc ?X_HB)
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        )
        :effect (and
            (empty ?arm)
            (not (clear ?lowerdisc))
            (not (athandpose ?arm ?disc ?X_HB))
            (atworldpose ?disc ?X_WB) 
            (on-disc ?disc ?lowerdisc)
        )
    )

    (:action unstack
        :parameters (?arm ?disc ?X_WB ?X_HB ?lowerdisc ?pre_q ?q)
        :precondition (and
            (arm ?arm)
            (disc ?disc)
            (disc ?lowerdisc)
            (smaller ?disc ?lowerdisc)
            (clear ?disc)
            (not (= ?disc ?lowerdisc))
            (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q)
            (atworldpose ?disc ?X_WB)
            (empty ?arm)
            (atconf ?arm ?pre_q)
            (on-disc ?disc ?lowerdisc)
            (worldpose ?disc ?X_WB)
            (handpose ?disc ?X_HB)
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        ) 
        :effect (and
            (athandpose ?arm ?disc ?X_HB)
            (clear ?lowerdisc)
            (not (atworldpose ?disc ?X_WB))
            (not (empty ?arm))
            (not (on-disc ?disc ?lowerdisc))
        )
    )
    
)