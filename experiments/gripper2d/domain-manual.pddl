(define (domain blocks_world)
    (:requirements :strips :derived-predicates :disjunctive-preconditions :equality)

    (:predicates
        (block ?block)
        (gripper ?gripper)
        (region ?region)
        (grasp ?gripper ?block ?grasp)
        (conf ?conf)
        (blockpose ?blockpose)
        (placement ?block ?region ?blockpose)

        (empty ?gripper)
        (atpose ?block ?pose)
        (on ?block ?region)
        (grasped ?gripper ?block ?grasped)
        (holding ?gripper ?block)
        (~safe-negative ?gripper ?conf ?block ?pose)
        (~safe-block-negative ?block1 ?pose1 ?block2 ?pose2)
        (ik ?gripper ?block ?conf ?X_WB ?X_HB)
    )

    (:action pick
        :parameters (?gripper ?block ?region ?X_WB ?X_HB ?conf)
        :precondition (and
            (empty ?gripper)
            ; (on ?block ?region)
            (atpose ?block ?X_WB)
            (grasp ?gripper ?block ?X_HB)
            (gripper ?gripper)
            (block ?block)
            (region ?region)
            (ik ?gripper ?block ?conf ?X_WB ?X_HB)
            ; (forall (?otherblock ?X_WB2) (imply (and (not (= ?block ?otherblock)) (atpose ?otherblock ?X_WB2)) (not (~safe-negative ?gripper ?conf ?otherblock ?X_WB2))))
        )
        :effect (and
            (not (empty ?gripper))
            (not (on ?block ?region))
            (not (atpose ?block ?X_WB))
            (grasped ?gripper ?block ?X_HB)
            (holding ?gripper ?block)
        )
    )

    (:action place
        :parameters (?gripper ?block ?region ?X_WB ?X_HB ?conf)
        :precondition (and
            (grasped ?gripper ?block ?X_HB)
            (grasp ?gripper ?block ?X_HB)
            (gripper ?gripper)
            (block ?block)
            (region ?region)
            (ik ?gripper ?block ?conf ?X_WB ?X_HB)
            (placement ?block ?region ?X_WB)
            ; (forall (?otherblock ?X_WB2) (imply (and (not (= ?block ?otherblock)) (atpose ?otherblock ?X_WB2)) (and (not (~safe-block-negative ?block ?X_WB ?otherblock ?X_WB2)) (not (~safe-negative ?gripper ?conf ?otherblock ?X_WB2)))))
        )
        :effect (and
            (empty ?gripper)
            (on ?block ?region)
            (atpose ?block ?X_WB)
            (not (grasped ?gripper ?block ?X_HB))
            (not (holding ?gripper ?block))
        )
    )

)