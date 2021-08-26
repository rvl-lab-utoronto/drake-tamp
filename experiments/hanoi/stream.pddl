(define (stream hanoi)

    (:stream find-traj
        :inputs (?arm ?q1 ?q2) 
        :fluents (atworldpose athandpose atconf)
        :domain (and
            (arm ?arm)
            (conf ?arm ?q1)  
            (conf ?arm ?q2)  
        )
        :outputs (?traj)
        :certified (and
            (motion ?arm ?q1 ?traj ?q2) 
        )
    )
    
    (:stream find-grasp
        :inputs (?disc)
        :domain (and
            (disc ?disc) 
        ) 
        :outputs (?X_HB)
        :certified (and
            (handpose ?disc ?X_HB)
        )
    )

    (:stream find-ik
        :inputs (?arm ?disc ?X_WB ?X_HB)
        :domain (and
            (arm ?arm)
            (disc ?disc)
            (worldpose ?disc ?X_WB) 
            (handpose ?disc ?X_HB)
        ) 
        :outputs (?pre_q ?q)
        :certified (and
            (ik ?arm ?disc ?X_WB ?X_HB ?pre_q ?q) 
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        )
    )

    ;(:stream check-colfreedisc
        ;:inputs (?arm ?q ?disc ?X_WB)
        ;:domain (and
            ;(arm ?arm)
            ;(disc ?disc) 
            ;(graspconf ?arm ?q)
            ;(worldpose ?disc ?X_WB)
        ;)
        ;:certified (and
            ;(colfree-disc ?arm ?q ?disc ?X_WB)    
        ;)
    ;)

    (:stream find-peg-place
        :inputs (?disc ?peg) 
        :domain (and
            (disc ?disc)
            (peg ?peg) 
        )
        :outputs (?X_WB)
        :certified (and
            (worldpose ?disc ?X_WB)    
            (peg-support ?disc ?X_WB ?peg)
        )
    )

    (:stream find-disc-place
        :inputs (?disc ?lowerdisc ?X_WL) 
        :domain (and
            (disc ?disc) 
            (disc ?lowerdisc) 
            (smaller ?disc ?lowerdisc)
            (worldpose ?lowerdisc ?X_WL)
        )
        :outputs (?X_WB)
        :certified (and
            (worldpose ?disc ?X_WB) 
            (disc-support ?disc ?X_WB ?lowerdisc ?X_WL)
        )
    )

)