(define (stream blocks_world)

    ; (:stream find-traj
    ;     :inputs (?arm ?q1 ?q2) 
    ;     :fluents (atworldpose athandpose atconf)
    ;     :domain (and
    ;         (arm ?arm)
    ;         (conf ?arm ?q1)  
    ;         (conf ?arm ?q2)  
    ;     )
    ;     :outputs (?traj)
    ;     :certified (and
    ;         (motion ?arm ?q1 ?traj ?q2) 
    ;     )
    ; )
    
    (:stream find-grasp
        :inputs (?block)
        :domain (and
            (block ?block) 
        ) 
        :outputs (?X_HB)
        :certified (and
            (handpose ?block ?X_HB)
        )
    )

    (:stream find-ik
        :inputs (?arm ?block ?X_WB ?X_HB)
        :domain (and
            (arm ?arm)
            (worldpose ?block ?X_WB) 
            (handpose ?block ?X_HB)
        ) 
        :outputs (?pre_q ?q)
        :certified (and
            (ik ?arm ?block ?X_WB ?X_HB ?pre_q ?q) 
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        )
    )

    (:stream check-colfree-block
        :inputs (?arm ?q ?block ?X_WB)
        :domain (and
            (arm ?arm)
            (block ?block) 
            (graspconf ?arm ?q)
            (worldpose ?block ?X_WB)
        )
        :certified (and
            (colfree-block ?arm ?q ?block ?X_WB)    
        )
    )

    (:stream find-table-place
        :inputs (?block ?table) 
        :domain (and
            (block ?block)
            (table ?table) 
        )
        :outputs (?X_WB)
        :certified (and
            (worldpose ?block ?X_WB)    
            (table-support ?block ?X_WB ?table)
        )
    )

    (:stream find-block-place
        :inputs (?block ?lowerblock ?X_WL) 
        :domain (and
            (block ?block) 
            (block ?lowerblock) 
            (worldpose ?lowerblock ?X_WL)
        )
        :outputs (?X_WB)
        :certified (and
            (worldpose ?block ?X_WB) 
            (block-support ?block ?X_WB ?lowerblock ?X_WL)
        )
    )

)