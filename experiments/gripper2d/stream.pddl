(define (stream blocks_world)

    (:stream grasp
        :inputs (?gripper ?block)
        :domain (and
            (gripper ?gripper)
            (block ?block)
        )
        :outputs (?grasp)
        :certified (and
            (grasp ?gripper ?block ?grasp)
        )
    )
    
    (:stream placement
        :inputs (?block ?region)
        :domain (and
            (region ?region)
            (block ?block)
        )
        :outputs (?placement)
        :certified (and
            (placement ?block ?region ?placement)
            (blockpose ?placement)
        )
    )
    (:stream ik
        :inputs (?gripper ?block ?blockpose ?grasp)
        :domain (and
            (gripper ?gripper)
            (block ?block)
            (blockpose ?blockpose)
            (grasp ?gripper ?block ?grasp)
        )
        :outputs (?conf)
        :certified (and
            (ik ?gripper ?block ?conf ?blockpose ?grasp)
            (conf ?conf)
        )
    )
    (:stream safe
        :inputs (?gripper ?conf ?block ?blockpose)
        :domain (and
            (gripper ?gripper)
            (block ?block)
            (blockpose ?blockpose)
            (conf ?conf)
        )
        :outputs ()
        :certified (and
            (safe ?gripper ?conf ?block ?blockpose)
        )
    )
    (:stream safe-block
        :inputs (?block1 ?blockpose1 ?block ?blockpose)
        :domain (and
            (block ?block1)
            (block ?block)
            (blockpose ?blockpose)
            (blockpose ?blockpose1)
        )
        :outputs ()
        :certified (and
            (safe-block ?block1 ?blockpose1 ?block ?blockpose)
        )
    )
)