(define (stream mega_domain)

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
        :inputs (?item)
        :domain (and
            (item ?item) 
        ) 
        :outputs (?X_HI)
        :certified (and
            (handpose ?item ?X_HI)
        )
    )

    (:stream find-ik
        :inputs (?arm ?item ?X_WI ?X_HI)
        :domain (and
            (arm ?arm)
            (worldpose ?item ?X_WI) 
            (handpose ?item ?X_HI)
        ) 
        :outputs (?pre_q ?q)
        :certified (and
            (ik ?arm ?item ?X_WI ?X_HI ?pre_q ?q) 
            (conf ?arm ?pre_q)
            (graspconf ?arm ?q)
        )
    )

    (:stream check-colfree-empty
        :inputs (?arm ?q ?item ?X_WI)
        :domain (and
            (arm ?arm)
            (item ?item) 
            (graspconf ?arm ?q)
            (worldpose ?item ?X_WI)
        )
        :certified (and
            (colfree-empty ?arm ?q ?item ?X_WI)    
        )
    )

    (:stream check-colfree-holding
        :inputs (?arm ?q ?item ?X_WI)
        :domain (and
            (arm ?arm)
            (item ?item) 
            (graspconf ?arm ?q)
            (worldpose ?item ?X_WI)
        )
        :certified (and
            (colfree-holding ?arm ?q ?item ?X_WI)    
        )
    )

    (:stream find-region-place
        :inputs (?item ?region) 
        :domain (and
            (item ?item)
            (region ?region) 
        )
        :outputs (?X_WI)
        :certified (and
            (worldpose ?item ?X_WI)    
            (region-support ?item ?X_WI ?region)
        )
    )

    (:stream find-item-place
        :inputs (?item ?loweritem ?X_WL) 
        :domain (and
            (item ?item) 
            (item ?loweritem) 
            (worldpose ?loweritem ?X_WL)
        )
        :outputs (?X_WI)
        :certified (and
            (worldpose ?item ?X_WI) 
            (item-support ?item ?X_WI ?loweritem ?X_WL)
        )
    )

)