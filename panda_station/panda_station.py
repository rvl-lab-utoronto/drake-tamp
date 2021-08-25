"""
A system used for TAMP experiments with the Franka Emika Panda arm and the Franka Emika Gripper
"""
import numpy as np
import pydrake.all
from . import construction_utils
from .panda_hand_position_controller import (
    PandaHandPositionController,
    make_multibody_state_to_panda_hand_state_system,
)

HAND_FRAME_NAME = "panda_hand"
ARM_LINK_PREFIX = "panda_link"

class PandaStation(pydrake.systems.framework.Diagram):
    """
    The PandaStation class
    TODO(agro): add cameras if nessecary
    """

    def __init__(self, time_step=0.001, name = "panda_station", dummy = False):
        """
        Construct a panda station

        Args:
            time_step: simulation time step [float]
            name: the name of the station
            dummy: if True, none of the ports are connnected
        """
        pydrake.systems.framework.Diagram.__init__(self)
        self.time_step = time_step
        self.dummy = dummy
        self.builder = pydrake.systems.framework.DiagramBuilder()
        (
            self.plant,
            self.scene_graph,
        ) = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
            self.builder, time_step=self.time_step
        )
        # dict in the form: {object_name: (ObjectInfo, Xinit_WO)}
        self.object_infos = {}  # list of tuples (ObjectInfo, Xinit_WO)
        self.directive = None  # the directive used to setup the environment
        self.plant.set_name("plant")
        self.set_name(name)

        # list of tuples of the form
        # (panda_model_index, hand_model_index, X_WB, name, weld_fingers)
        self.panda_infos = {}
        self.frame_groups = {}

    def fix_collisions(self):
        """
        fix collisions for the hand and arm in this plant
        by removing collisions between panda_link5<->panda_link7 and
        panda_link7<->panda_hand

        Then remove collisions within each frame group
        """
        assert len(self.panda_infos) != 0, "No panda has been added"
        # get geometry indices and create geometry sets
        
        for info in self.panda_infos.values():
            hand = info.hand
            panda = info.panda
            link5 = self.plant.GetBodyByName(ARM_LINK_PREFIX + "5", panda)
            link7 = self.plant.GetBodyByName(ARM_LINK_PREFIX + "7", panda)
            link6 = self.plant.GetBodyByName(ARM_LINK_PREFIX + "6", panda)
            link8 = self.plant.GetBodyByName(ARM_LINK_PREFIX + "8", panda)
            hand = self.plant.GetBodyByName(HAND_FRAME_NAME, hand)

            l57_set = self.plant.CollectRegisteredGeometries([link5, link7])
            self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(l57_set))
            lh7_set = self.plant.CollectRegisteredGeometries([hand, link7])
            self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(lh7_set))
            #lh8_set = self.plant.CollectRegisteredGeometries([hand, link8])
            #self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(lh8_set))
            l68_set = self.plant.CollectRegisteredGeometries([link6, link8])
            self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(l68_set))

        for frame in self.frame_groups:
            if frame.name() == HAND_FRAME_NAME:
                for obj_id in self.frame_groups[frame]:
                    self.remove_collisions_with_hands(obj_id)

            bodies = []
            for object_info in self.frame_groups[frame]:
                body_infos = list(object_info.get_body_infos().values())
                for body_info in body_infos:
                    bodies.append(body_info.get_body())
            geom_set = self.plant.CollectRegisteredGeometries(bodies)
            self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(geom_set))

    def remove_collisions_with_hands(self, object_info):
        """
        Removes collisions between all links of the object in object_info
        and the panda hand
        """
        body_infos = list(object_info.get_body_infos().values())
        obj_bodies = []
        for body_info in body_infos:
            obj_bodies.append(body_info.get_body())

        for panda_info in self.panda_infos.values():
            hand = panda_info.hand
            hand_body_ids = self.plant.GetBodyIndices(hand)
            bodies = [self.plant.get_body(id) for id in hand_body_ids] + obj_bodies
            geom_set = self.plant.CollectRegisteredGeometries(bodies)
            self.scene_graph.collision_filter_manager().Apply(pydrake.geometry.CollisionFilterDeclaration().ExcludeWithin(geom_set))

        """
        hand_col_ids = []
        for body_id in hand_body_ids:
            hand_col_ids += self.plant.GetCollisionGeometriesForBody(
                self.plant.get_body(body_id)
            )
        hand_geom_set = pydrake.geometry.GeometrySet(hand_col_ids)
        body_infos = list(object_info.get_body_infos().values())
        body_col_ids = []
        for body_info in body_infos:
            body_col_ids += self.plant.GetCollisionGeometriesForBody(
                body_info.get_body()
            )
        body_geom_set = pydrake.geometry.GeometrySet(body_col_ids)
        self.scene_graph.ExcludeCollisionsBetween(hand_geom_set, body_geom_set)
        """

    def add_panda_with_hand(
        self,
        weld_fingers=False,
        blocked = False,
        q_initial=np.array([0.0, 0.1, 0, -1.2, 0, 1.6, np.pi/4]),
        X_WB = pydrake.math.RigidTransform(),
        panda_name = None,
        hand_name = None
    ):
        """
        Add the panda hand and panda arm (at the world origin) to the station

        Args:
            weld_fingers: [bool] True iff the fingers are welded in the open position
            q_initial: the initial positions to set the panda arm (np.array)
        """
        if panda_name is None:
            panda_name = "panda" + str(len(self.panda_infos))
        if hand_name is None:
            hand_name = "hand" + str(len(self.panda_infos))
        panda = construction_utils.add_panda(
            self.plant,
            q_initial=q_initial,
            X_WB = X_WB,
            name = panda_name
        )
        hand = construction_utils.add_panda_hand(
            self.plant,
            panda_model_instance_index=panda,
            weld_fingers=weld_fingers,
            blocked = blocked,
            name = hand_name,
        )
        self.panda_infos[panda_name] = PandaInfo(
            panda, hand, panda_name, hand_name, X_WB, weld_fingers or blocked
        )
        return self.panda_infos[panda_name]

    def setup_from_file(self, filename, names_and_links=None):
        """
        Setup a station with the path to the directive in `filename`

        Args:
            filename: [string] that path (starting from this directory of this
            file) to the directive. e.g `directives/table_top.yaml`

            names_name_links: provide a list of the form
            [(model_name, main_link_name), ... ]
            to be added to this PandaStations object_infos.
            This is used if you want these objects to be considerd for placement
        """
        self.directive = construction_utils.find_resource(filename)
        parser = pydrake.multibody.parsing.Parser(self.plant)
        construction_utils.add_package_paths(parser)
        pydrake.multibody.parsing.ProcessModelDirectives(
            pydrake.multibody.parsing.LoadModelDirectives(self.directive),
            self.plant,
            parser,
        )
        if names_and_links is not None:
            for name, link_name in names_and_links:
                object_info = ObjectInfo(name)
                model = self.plant.GetModelInstanceByName(name)
                body_indices = self.plant.GetBodyIndices(model)
                for i in body_indices:
                    body = self.plant.get_body(i)
                    body_info = BodyInfo(body, i)
                    if body.name() == link_name:
                        object_info.set_main_body_info(body_info)
                    else:
                        object_info.add_body_info(body_info)
                self.object_infos[name] = (object_info, None)

    def add_model_from_file(
        self, path, Xinit_PO, P=None, welded=False, main_body_name=None, name=None
    ):
        """
        Add a model to this plant from the full path provided in `path`
        at initial position Xinit_PO relative to parent frame P (default to
        world frame)

        Args:
            path: [string] full path to model file (eg. from FindResourceOrThrow or
            find_resource)
            Xinit_PO: the initial object pose relative to the parent frame P
            P: parent frame (defaults to world frame)
            main_body_name: [string] provide the name of the body link to set the
            position of the model, if there is more than one link in the model.
            name: [string] optional name for the model.
            welded: True iff the body should be welded to a fixedoffsetframe
            attached to the parent frame
        Returns:
            the model instance index of the added model
        """
        parser = pydrake.multibody.parsing.Parser(self.plant)
        if name is None:
            num = str(len(self.object_infos))
            name = "added_model_" + num
        model = parser.AddModelFromFile(path, name)
        indices = self.plant.GetBodyIndices(model)
        assert (len(indices) == 1) or (
            main_body_name is not None
        ), "You must specify the main link name"
        main_body_index = indices[0]
        body_infos = {}
        if main_body_name is not None:
            for i in indices:
                test_name = self.plant.get_body(i).name()
                body_infos[i] = BodyInfo(self.plant.get_body(i), i)
                if test_name == main_body_name:
                    main_body_index = i
        Xinit_WO = None
        if P is None:
            P = self.plant.world_frame()
            Xinit_WO = Xinit_PO  # assume parent frame is world frame
        offset_frame = None
        main_body = self.plant.get_body(main_body_index)
        if welded:
            frame_name = "offset_frame_" + name
            offset_frame = pydrake.multibody.tree.FixedOffsetFrame(
                frame_name, P=P, X_PF=Xinit_PO
            )
            self.plant.AddFrame(offset_frame)
            self.plant.WeldFrames(
                offset_frame, main_body.body_frame(), pydrake.math.RigidTransform()
            )
        main_body_info = BodyInfo(main_body, main_body_index)
        object_info = ObjectInfo(name, welded_to_frame=offset_frame, path=path)
        object_info.body_infos = body_infos  # add all the other body infos
        object_info.set_main_body_info(main_body_info)
        self.object_infos[name] = (object_info, Xinit_WO)
        if welded:
            if P not in self.frame_groups:
                self.frame_groups[P] = [object_info]
            else:
                self.frame_groups[P].append(object_info)
        return object_info


    def get_directive(self):
        """
        Return the directive used to create the static objects in this plant
        """
        return self.directive

    def get_multibody_plant(self):
        """
        Returns the multibody plant of this panda station
        """
        return self.plant

    def get_scene_graph(self):
        """
        Returns the scene graph of this panda station
        """
        return self.scene_graph

    def get_plant_and_scene_graph(self):
        """
        Returns the plant and scene graph of this panda station
        """
        return self.plant, self.scene_graph

    def get_panda(self):
        """
        Returns the panda arm ModelInstanceIndex of this station
        """
        assert len(self.panda_infos) == 1, f"There are {len(self.panda_infos)} pandas in this station"
        return list(self.panda_infos.values())[0].panda

    def get_hand(self):
        """
        Returns the panda hand ModelInstanceIndex of this station
        """
        assert len(self.panda_infos) == 1, f"There are {len(self.panda_infos)} pandas in this station"
        return list(self.panda_infos.values())[0].hand

    def get_panda_joint_limits(self):
        """
        Get the joint limits for the panda in the form:
        [
        (lower, upper),
        ...
        ]
        in radians
        """
        assert len(self.panda_infos) > 0
        panda = list(self.panda_infos.values())[0].panda
        num_q = self.plant.num_positions(panda)
        joint_inds = self.plant.GetJointIndices(panda)[:num_q]
        joint_limits = []
        for i in joint_inds:
            joint = self.plant.get_joint(i)
            joint_limits.append(
                (joint.position_lower_limits()[0], joint.position_upper_limits()[0])
            )
        return joint_limits

    def get_panda_lower_limits(self):
        """
        Get the lower limits of the panda in a numpy array
        """
        assert len(self.panda_infos) > 0
        panda = list(self.panda_infos.values())[0].panda
        num_q = self.plant.num_positions(panda)
        joint_inds = self.plant.GetJointIndices(panda)[:num_q]
        joint_limits = []
        for i in joint_inds:
            joint = self.plant.get_joint(i)
            joint_limits.append(joint.position_lower_limits()[0])
        return np.array(joint_limits)

    def get_panda_upper_limits(self):
        """
        Get the upper limits of the panda in a numpy array
        """
        assert len(self.panda_infos) > 0
        panda = list(self.panda_infos.values())[0].panda
        num_q = self.plant.num_positions(panda)
        joint_inds = self.plant.GetJointIndices(panda)[:num_q]
        joint_limits = []
        for i in joint_inds:
            joint = self.plant.get_joint(i)
            joint_limits.append(joint.position_upper_limits()[0])
        return np.array(joint_limits)

    def finalize(self):
        """finalize the panda station"""


        assert len(self.panda_infos) != 0, "No panda added, run add_panda_with_hand"

        inspector = self.scene_graph.model_inspector()
        counter = 0

        for object_info, _ in self.object_infos.values():
            for body_info in object_info.body_infos.values():
                for i, geom_id in enumerate(
                    self.plant.GetCollisionGeometriesForBody(body_info.get_body())
                ):
                    shape = inspector.GetShape(geom_id)
                    X_BG = inspector.GetPoseInFrame(geom_id)
                    frame_name = (
                        "frame_"
                        + object_info.get_name()
                        + "_"
                        + body_info.get_name()
                        + "_"
                        + str(counter)
                    )
                    counter += 1 # to ensure unique names
                    frame = self.plant.AddFrame(
                        pydrake.multibody.tree.FixedOffsetFrame(
                            frame_name, body_info.get_body_frame(), X_BG
                        )
                    )
                    body_info.add_shape_info(ShapeInfo(shape, frame))

        self.plant.Finalize()
        self.fix_collisions()

        for object_info, Xinit_WO in self.object_infos.values():
            if Xinit_WO is None:
                # the object is welded
                continue
            main_body = object_info.main_body_info.get_body()
            self.plant.SetDefaultFreeBodyPose(main_body, Xinit_WO)

        if self.dummy:
            infos = []
        else:
            infos = self.panda_infos.values()

        for info in infos:
            panda, hand, panda_name, hand_name, X_WB, weld_fingers = info.unpack()

            num_panda_positions = self.plant.num_positions(panda)

            panda_position = self.builder.AddSystem(
                pydrake.systems.primitives.PassThrough(num_panda_positions)
            )
            info.add_panda_position_port(
                self.builder.ExportInput(
                    panda_position.get_input_port(),
                    f"{panda_name}_position"
                )
            )
            self.builder.ExportOutput(
                panda_position.get_output_port(),
                f"{panda_name}_position_command"
            )

            demux = self.builder.AddSystem(
                pydrake.systems.primitives.Demultiplexer(
                    2 * num_panda_positions, num_panda_positions
                )
            )
            self.builder.Connect(
                self.plant.get_state_output_port(panda), demux.get_input_port()
            )
            info.add_panda_position_measured_port(
                self.builder.ExportOutput(
                    demux.get_output_port(0),
                    f"{panda_name}_position_measured"
                )
            )
            self.builder.ExportOutput(
                demux.get_output_port(1),
                f"{panda_name}_velocity_estimated"
            )
            self.builder.ExportOutput(
                self.plant.get_state_output_port(panda),
                f"{panda_name}_state_estimated"
            )

            controller_plant = pydrake.multibody.plant.MultibodyPlant(
                time_step=self.time_step
            )
            # plant for the panda controller
            controller_panda = construction_utils.add_panda(
                controller_plant,
                X_WB = X_WB,
                name = panda_name
            )
            # welded so the controller doesn't care about the hand joints
            construction_utils.add_panda_hand(
                controller_plant,
                panda_model_instance_index=controller_panda,
                weld_fingers=True,
            )
            controller_plant.Finalize()

            panda_controller = self.builder.AddSystem(
                pydrake.systems.controllers.InverseDynamicsController(
                    controller_plant,
                    kp=[100] * num_panda_positions,
                    ki=[1] * num_panda_positions,
                    kd=[20] * num_panda_positions,
                    has_reference_acceleration=False,
                )
            )

            panda_controller.set_name(f"{panda_name}_controller")
            self.builder.Connect(
                self.plant.get_state_output_port(panda),
                panda_controller.get_input_port_estimated_state(),
            )
            # feedforward torque
            adder = self.builder.AddSystem(
                pydrake.systems.primitives.Adder(2, num_panda_positions)
            )
            self.builder.Connect(
                panda_controller.get_output_port_control(),
                adder.get_input_port(0)
            )
            # passthrough to make the feedforward torque optional (default to zero values)
            torque_passthrough = self.builder.AddSystem(
                pydrake.systems.primitives.PassThrough([0] * num_panda_positions)
            )
            self.builder.Connect(
                torque_passthrough.get_output_port(),
                adder.get_input_port(1)
            )
            self.builder.ExportInput(
                torque_passthrough.get_input_port(),
                f"{panda_name}_feedforward_torque"
            )
            self.builder.Connect(
                adder.get_output_port(),
                self.plant.get_actuation_input_port(panda)
            )

            # add a discete derivative to find velocity command based on positional commands
            desired_state_from_position = self.builder.AddSystem(
                pydrake.systems.primitives.StateInterpolatorWithDiscreteDerivative(
                    num_panda_positions, self.time_step, suppress_initial_transient=True
                )
            )
            desired_state_from_position.set_name(f"desired_{panda_name}_state_from_position")
            self.builder.Connect(
                desired_state_from_position.get_output_port(),
                panda_controller.get_input_port_desired_state(),
            )
            self.builder.Connect(
                panda_position.get_output_port(),
                desired_state_from_position.get_input_port(),
            )

            if not weld_fingers:
                # TODO(agro): make sure this hand controller is accurate
                hand_controller = self.builder.AddSystem(PandaHandPositionController())
                hand_controller.set_name(f"{hand_name}_controller")
                self.builder.Connect(
                    hand_controller.GetOutputPort(f"generalized_force"),
                    self.plant.get_actuation_input_port(hand),
                )
                self.builder.Connect(
                    self.plant.get_state_output_port(hand),
                    hand_controller.GetInputPort("state"),
                )
                info.add_hand_position_port(
                    self.builder.ExportInput(
                        hand_controller.GetInputPort("desired_position"),
                        f"{hand_name}_position"
                    )
                )
                self.builder.ExportInput(
                    hand_controller.GetInputPort("force_limit"),
                    f"{hand_name}_force_limit"
                )
                hand_mbp_state_to_hand_state = self.builder.AddSystem(
                    make_multibody_state_to_panda_hand_state_system()
                )
                self.builder.Connect(
                    self.plant.get_state_output_port(hand),
                    hand_mbp_state_to_hand_state.get_input_port(),
                )
                info.add_hand_state_measured_port(
                    self.builder.ExportOutput(
                        hand_mbp_state_to_hand_state.get_output_port(),
                        f"{hand_name}_state_measured"
                    )
                )
                self.builder.ExportOutput(
                    hand_controller.GetOutputPort("grip_force"),
                    f"{hand_name}_force_measured"
                )

        # TODO(agro): cameras if needed

        # export cheat ports
        self.builder.ExportOutput(
            self.scene_graph.get_query_output_port(), "geometry_query"
        )
        self.builder.ExportOutput(
            self.plant.get_contact_results_output_port(), "contact_results"
        )
        self.builder.ExportOutput(
            self.plant.get_state_output_port(), "plant_continuous_state"
        )

        # for visualization

        self.builder.ExportOutput(
            self.scene_graph.get_query_output_port(), "query_object"
        )

        self.builder.BuildInto(self)


class ObjectInfo:
    """
    Class for storing all bodies associated with an object
    """

    def __init__(self, name, welded_to_frame=None, path=None):
        """
        Construct an ObjectInfo object

        Args:
            welded_to_frame: the FixedOffsetFrame that the main body is welded
            to (optional)
            path: the absolute filepath used to find the model (optional)
        """
        self.path = path
        self.main_body_info = None
        self.welded_to_frame = welded_to_frame
        self.body_infos = {}
        self.name = name
        # which shapes are suitable for grasping
        self.graspable_shapes = []
        self.placeable_shapes = []
        # the Surface instances of the shapes that objects can be placed on
        self.surfaces = {}

    def query_shape_infos(self, criteria_func = None):
        """
        Return all ShapeInfos of this ObjectInfo that 
        satisfy criteria_func, with the signature:

        def criteria_func(shape_info):
            # returns boolean

        which will specify whether a shape_info is included in the
        output of this query
        """
        shape_infos = []
        body_infos = list(self.get_body_infos().values())
        for body_info in body_infos:
            for shape_info in body_info.get_shape_infos():
                if (criteria_func is None) or (criteria_func(shape_info)):
                    shape_infos.append(shape_info)
        return shape_infos

    def set_main_body_info(self, main_body_info):
        """
        Set the main body info of this ObjectInfo
        """
        self.main_body_info = main_body_info
        if not main_body_info.get_index() in self.body_infos.keys():
            self.body_infos[main_body_info.get_index()] = self.main_body_info

    def add_body_info(self, body_info):
        """
        Given a BodyInfo, add it to this object
        """
        self.body_infos[body_info.get_index()] = body_info

    def get_main_body(self):
        """
        Return the BodyInfo of the main body of this object
        """
        return self.main_body_info

    def get_frame(self):
        """
        Get the welded_to_frame
        """
        return self.welded_to_frame

    def get_path(self):
        """
        returns the path used to create this object
        """
        return self.path

    def get_name(self):
        """
        return the name of this ObjectInfo
        """
        return self.name

    def get_body_infos(self):
        """
        Return the body infos associated with this object
        """
        return self.body_infos

    def str_info(self):
        """
        get string info about this ObjectInfo
        """
        frame_name = None
        if self.welded_to_frame is not None:
            frame_name = self.welded_to_frame.name()
        res = f"""
        name: {self.name}
        path: {self.path}
        main body name: {self.main_body_info.get_name()}
        welded to frame: {frame_name}
        BodyInfos: 
        """
        for info in self.body_infos.values():
            str_info = str(info).split("\n")
            for string in str_info:
                res += "\t" + string + "\n"

        return res

    def __str__(self):
        res = f"Object name: {self.name}"
        return res


class BodyInfo:
    """
    Class for storing all geometries associated with a body
    """

    def __init__(self, body, body_index):
        """
        Construct a body info instance by providing its
        pydrake.multibody.tree.BodyIndex

        Args:
            body_index: the BodyIndex of the main body
            welded to (if it is welded to one)
            body: the actual Body associated with this body
            index
        """
        self.body_index = body_index
        self.body = body
        self.shape_infos = []

    def add_shape_info(self, shape_info):
        """
        Add a ShapeInfo instance to be associated with this body
        """
        self.shape_infos.append(shape_info)

    def get_name(self):
        """
        Get the name of this body
        """
        return self.body.name()

    def get_index(self):
        """
        Return the index of this body
        """
        return self.body.index()

    def get_body_frame(self):
        """
        Return the body frame of this body
        """
        return self.body.body_frame()

    def get_body(self):
        """
        Returns this body
        """
        return self.body

    def get_shape_infos(self):
        """
        Return the shape infos associated with this body
        """
        return self.shape_infos

    def __str__(self):
        res = f"""body index: {self.body_index}, body_name: {self.body.name()}
        ShapeInfos: 
        """

        for info in self.shape_infos:
            str_info = str(info).split("\n")
            for string in str_info:
                res += "\t" + string + "\n\t"

        return res


class ShapeInfo:
    """
    Class for storing the information about a shape
    """

    def __init__(self, shape, offset_frame):

        """
        Construct a ShapeInfo instance given a
        pydrake.geometry.GeometryInstance.shape and its
        associated pydrake.multibody.tree.FixedOffsetFrame
        """
        self.shape = shape
        self.offset_frame = offset_frame
        self.type = type(shape)

    def __str__(self):
        """
        Used when printing out the shape info
        """
        res = f"Offset_frame: {self.offset_frame.name()}, type:"
        if self.type == pydrake.geometry.Box:
            res += "box"
        if self.type == pydrake.geometry.Cylinder:
            res += "cylinder"
        if self.type == pydrake.geometry.Sphere:
            res += "sphere"
        return res

class PandaInfo:

    def __init__(
        self,
        panda,
        hand,
        panda_name,
        hand_name,
        X_WB,
        weld_fingers):

        self.panda = panda
        self.hand = hand
        self.panda_name = panda_name
        self.hand_name = hand_name
        self.X_WB = X_WB
        self.weld_fingers = weld_fingers
        self.ports = {}

    def unpack(self):
        return (
            self.panda,
            self.hand,
            self.panda_name,
            self.hand_name,
            self.X_WB,
            self.weld_fingers
        )

    def add_panda_position_port(self, port_index):
        self.ports["panda_position"] = port_index

    def get_panda_position_port(self):
        return self.ports["panda_position"]

    def add_panda_position_measured_port(self, port_index):
        self.ports["panda_position_measured"] = port_index

    def get_panda_position_measured_port(self):
        return self.ports["panda_position_measured"]

    def add_hand_state_measured_port(self, port_index):
        self.ports["hand_state_measured"] = port_index

    def get_hand_state_measured_port(self):
        return self.ports["hand_state_measured"]

    def add_hand_position_port(self, port_index):
        self.ports["hand_position"] = port_index

    def get_hand_position_port(self):
        return self.ports["hand_position"]