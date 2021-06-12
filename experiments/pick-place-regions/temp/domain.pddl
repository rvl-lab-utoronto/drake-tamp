(define (domain pickplaceregions)
    (:requirements :strips)
    (:predicates
        ; discret types
        (item ?item) ; item name
        (arm ?a) ; robotarm (in case we have more than one)
        (region ?r) ; e.g. table, red, green

        ; continuous types
        (pose ?item ?pose) ; a valid pose of an item
        (relpose ?item ?grasppose); a pose relative to the hand
        (conf ?conf) ; robot configuration
        (contained ?item ?region ?pose) ; if ?item were at ?pose, it would be inside ?region

        (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)
        (place ?item ?region ?grasppose ?placementpose ?placeconf ?preplaceconf ?postplaceconf)
        (mftraj ?traj ?start ?end)
        (mhtraj ?item ?startconf ?endconf ?traj)

        ;fluents
        (at ?arm ?conf)
        (empty ?arm)
        (grasped ?arm ?item)
        (atpose ?item ?pose)
        (atgrasppose ?item ?grasppose)

        ; derived
        (in ?item ?region)
    )
    (:action move-free
        :parameters (?arm ?start ?end ?t)
        :precondition (and
            (arm ?arm)
            (conf ?start)
            (conf ?end)
            (empty ?arm)
            (at ?arm ?start)
            (mftraj ?t ?start ?end)
        )
        :effect (and
            (not (at ?arm ?start)) (at ?arm ?end))
    )
    (:action pick
        :parameters (?arm ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf) ; grasppose is X_Hand_Item
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?pregraspconf)
            (conf ?postgraspconf)
            (conf ?graspconf)
            (pose ?item ?pose)
            (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)

            (empty ?arm)
            (atpose ?item ?pose)
            (at ?arm ?pregraspconf)
        )
        :effect (and
            (not (empty ?arm))
            (not (at ?arm ?pregraspconf))
            (not (atpose ?item ?pose))

            (grasped ?arm ?item)
            (at ?arm ?postgraspconf)
            (atgrasppose ?item ?grasppose)
        )
    )
    (:action move-holding
        :parameters (?arm ?item ?startconf ?endconf ?t)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?startconf)
            (conf ?endconf)
            (at ?arm ?startconf)
            (grasped ?arm ?item)
            (mhtraj ?item ?startconf ?endconf ?t)
        )
        :effect (and
            (not (at ?arm ?startconf))
            (at ?arm ?endconf)
        )
    )
    (:action place
        :parameters (?arm ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (region ?region)
            (pose ?item ?placepose)
            (conf ?preplaceconf)
            (conf ?postplaceconf)
            (conf ?placeconf)
            (place ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
            

            (at ?arm ?preplaceconf)
            (grasped ?arm ?item)
        )
        :effect (and
            (not (grasped ?arm ?item))
            (not (at ?arm ?preplaceconf))
            (not (atgrasppose ?item ?grasppose))

            (empty ?arm)
            (at ?arm ?preplaceconf)
            (atpose ?item ?placepose)
        )
    )
    (:derived (in ?item ?region)
        (exists
            (?pose)
            (and
                (pose ?item ?pose)
                (region ?region)
                (item ?item)
                (contained ?item ?region ?pose)
                (atpose ?item ?pose)
            )
        )
    )
)