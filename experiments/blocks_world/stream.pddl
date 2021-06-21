(define (stream kitchen)

    ;(:function (distance ?traj)
        ;(traj ?traj)
    ;)

    (:stream find-traj
        :inputs (?q1 ?q2) 
        :fluents (atpose holding)
        :domain (and
            (conf ?q1) 
            (conf ?q2)
        )
        :outputs (?traj)
        :certified (and
            (motion ?q1 ?traj ?q2) 
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

    (:stream find-place
        :inputs (?item ?region) 
        :domain (and
            (item ?item) 
            (region ?region)
        ) 
        :outputs (?X_WI)
        :certified (and 
            (contained ?item ?X_WI ?region)
            (worldpose ?item ?X_WI)
        )
    )

    (:stream find-ik
        :inputs (?item ?X_WI ?X_HI) 
        :domain (and
            (worldpose ?item ?X_WI)
            (handpose ?item ?X_HI)
        ) 
        :outputs (?pre_q ?q)
        :certified (and 
            (ik ?item ?X_WI ?X_HI ?pre_q ?q)
            (conf ?pre_q)
            (graspconf ?q)
        )
    )

    (:stream check-safe
        :inputs (?q ?item ?X_WI) 
        :domain (and
            (item ?item)
            (graspconf ?q)
            (worldpose ?item ?X_WI)
        )
        :certified (and
            (colfree ?q ?item ?X_WI) 
        )
    )

    

    ;(:stream check-safe-place
    ;    :inputs (?q ?itemholding ?X_HI ?item ?X_WI) 
    ;    :domain (and
    ;        (item ?itemholding)
    ;        (item ?item)
    ;        (graspconf ?q)
    ;        (worldpose ?item ?X_WI)
    ;        (handpose ?itemholding ?X_HI)
    ;    )
    ;    :certified (and
    ;        (colfreeholding ?q ?itemholding ?X_HI ?item ?X_WI) 
    ;    )
    ;)


    ;(:stream check-pick
    ;    :inputs (?grasp_q ?item ?X_WI)
    ;    :domain (and
    ;        (item ?item) 
    ;        (conf ?grasp_q)
    ;        (worldpose ?item ?X_WI)
    ;    ) 
    ;    :certified (safeikfree ?grasp_q ?item ?X_WI)
    ;)

    ;(:stream check-place
    ;    :inputs (?place_q ?holdingitem ?X_HI ?worlditem ?X_WI)
    ;    :domain (and
    ;        (item ?holdingitem) 
    ;        (item ?worlditem)
    ;        (conf ?place_q)
    ;        (worldpose ?worlditem ?X_WI)
    ;        (handpose ?holdingitem ?X_HI)
    ;    ) 
    ;    :certified (safeikholding ?place_q ?holdingitem ?X_HI ?worlditem ?X_WI)
    ;)
)