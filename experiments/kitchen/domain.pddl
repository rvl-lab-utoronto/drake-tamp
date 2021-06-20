(define (domain kitchen)
    (:requirements :strips :derived-predicates)
    (:predicates 
        ;discrete types
        (item ?item)  ;string
        (arm ?a) ;string
        (region ?r) ; a tuple of the form (object_name, link_name)
        (sink ?region)
        (burner ?region)

        ;continuous types
        (pose ?item ?pose)
        (relpose ?item ?grasppose)
        (conf ?conf)
        (graspconf ?conf)
        (contained ?item ?region ?pose) ; if item were at post, would it be inside region?

        ;stream certified predicates
        (grasp ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf)
        (place ?item ?region ?grasppose ?placementpose ?placeconf ?preplaceconf ?postplaceconf)
        (mftraj ?traj ?start ?end)
        (mhtraj ?item ?startconf ?endconf ?grasppose ?traj)

        ;fluents
        (at ?arm ?conf)
        (empty ?arm)
        (grasped ?arm ?item)
        (atpose ?item ?pose)
        (atgrasppose ?item ?grasppose)
        (clean ?item)
        (cooked ?item)

        ; derived
        (in ?item ?region)
    )

    (:action pick
        :parameters (?arm ?item ?pose ?grasppose ?graspconf ?pregraspconf ?postgraspconf) ; grasppose is X_Hand_Item
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?pregraspconf)
            (conf ?postgraspconf)
            (graspconf ?graspconf)
            (pose ?item ?pose)
            (relpose ?item ?grasppose)
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

    (:action move-holding
        :parameters (?arm ?item ?startconf ?endconf ?grasppose ?t)
        :precondition (and
            (arm ?arm)
            (item ?item)
            (conf ?startconf)
            (conf ?endconf)
            (at ?arm ?startconf)
            (grasped ?arm ?item)
            ;(atgrasppose ?item ?grasppose)
            ;(relpose ?item ?grasppose)
            (mhtraj ?item ?startconf ?endconf ?grasppose ?t)
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
            (graspconf ?placeconf)
            ;(relpose ?item ?grasppose)
            (place ?item ?region ?grasppose ?placepose ?placeconf ?preplaceconf ?postplaceconf)
            
            ;(atgrasppose ?item ?grasppose)
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

    (:action wash
        :parameters (?item ?region) 
        :precondition (and
            (region ?region)
            (item ?item) 
            (sink ?region) 
            (in ?item ?region) 
        )
        :effect (and
            (clean ?item)
        )
    )

    (:action cook
        :parameters (?item ?region) 
        :precondition (and 
            (item ?item)
            (region ?region)
            (burner ?region)
            (clean ?item)
            (in ?item ?region)
        )
        :effect (and
            (cooked ?item)
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