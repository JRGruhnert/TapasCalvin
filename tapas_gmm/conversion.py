import numpy as np
import torch
from tapas_gmm.utils.observation import (
    SceneObservation,
    SingleCamObservation,
    CameraOrder,
    dict_to_tensordict,
    empty_batchsize,
)


"""
Observation: {
    'rgb_obs': {}, 
    'depth_obs': {}, 
    'robot_obs': array(
        [ 0.0258664 , -0.23131439,  0.57127922,  3.09045976, -0.02907239,
        1.50013581,  0.07999752, -1.21778727,  1.03987546,  2.11978223,
       -2.34205014, -0.87015669,  1.6411785 ,  0.55345239,  1.        ]), 
    'scene_obs': array(
        [ 5.33374207e-04,  0.00000000e+00,  0.00000000e+00,  1.15648232e-19,
        0.00000000e+00,  0.00000000e+00,  2.64898695e-04,  6.43672906e-02,
        4.58565328e-01,  1.00525318e-02,  4.34458591e-03,  6.21360433e-01,
       -2.01627481e-01,  9.37904257e-02,  4.59609699e-01, -1.22418926e-02,
       -2.74741610e-03,  2.35725010e+00, -2.04815930e-01,  5.68664407e-02,
        4.59604075e-01,  1.24817618e-02,  2.59154692e-03, -5.00935321e-01])}
Reward: 0
Done: False
Info: {
'robot_info': {
    'tcp_pos': (0.025866400485963403, -0.23131438762118722, 0.5712792192094824), 
    'tcp_orn': (3.0904597641953355, -0.029072393606107024, 1.5001358134705938), 
    'gripper_opening_width': 0.07999751738846907, 
    'arm_joint_states': [-1.2177872689573035, 1.0398754601309899, 2.1197822349031354, -2.342050137154223, -0.8701566919837812, 1.6411784962514722, 0.5534523935873618], 
    'gripper_action': 1.0, 
    'uid': 0, 
    'contacts': ()}, 
'scene_info': {
    'fixed_objects': {
        'table': {
            'uid': 5, 
            'links': {
                'button_link': 0, 
                'switch_link': 1, 
                'slide_link': 2, 
                'drawer_link': 3, 
                'led_link': 4, 
                'light_link': 5, 
                'plank_link': 6, 
                'base_link': -1}, 
            'contacts': ((0, 5, 4, 6, -1, (-0.17956642860894878, 0.0222795317886946, 0.44099975154699544), (-0.17956642860895045, 0.022279538702395053, 0.4391175533707221), (-8.835466657234391e-13, 3.6732053678564587e-06, -0.9999999999932537), -0.0018821981762860286, 2.2834777569755933, -0.7908843972595445, (0.0, 0.9999999999932537, 3.6732053678564587e-06), -0.11510876087503996, (1.0, 3.2454483552870136e-18, -8.835466657174787e-13)), 
                (0, 5, 3, 6, -1, (-0.17500839102701984, 0.09303454000677441, 0.4410000114446528), 
                    (-0.17500839102701987, 0.09303454483742081, 0.4396849075341994), 
                    (-2.5328272455862733e-14, 3.6732051044613095e-06, -0.9999999999932537), 
                    -0.0013151039104622355, 7.31624108432548, 1.9810613629753318, 
                    (0.0, 0.9999999999932537, 3.6732051044613095e-06), 1.622498187244873, 
                    (1.0, 9.30359396720618e-20, -2.5328272455691866e-14)), (0, 5, 2, 2, -1, 
                    (-0.01062675516525527, 0.040823159321478585, 0.4728855755330677), (-0.01062675516525527, 0.031191158633994506, 0.4784466137228214), (0.0, -0.8660254037844348, 0.5000000000000068), -0.01112207637950723, 0.0, 0.0, (1.0, 0.0, 0.0), 0.0, (-0.0, 0.5000000000000068, 0.8660254037844347)), (0, 5, 2, 6, -1, (0.010855557565985191, 0.0941805464820626, 0.4410000156541696), (0.010855557565985191, 0.0941805559064174, 0.43843431226185786), (0.0, 3.673205102845747e-06, -0.9999999999932538), -0.0025657033923290225, 9.234945622780819, 3.228936004524131, (0.0, 0.9999999999932538, 3.673205102845747e-06), 0.14590856387912304, (1.0, -0.0, 0.0)))}}, 
    'movable_objects': {
        'block_red': {
            'current_pos': (0.00026489869479721484, 0.0643672905913041, 0.458565327933055), 
            'current_orn': (0.004121531353998952, 0.0036048198588842567, 0.30569138040784033, 0.9521149080848551), 
            'current_lin_vel': (0.03611939869843228, 0.29962138942949995, -0.03003241422727073), 
            'current_ang_vel': (3.0975413527520623, 1.5947505747614843, -0.7783795417452268), 
            'contacts': ((0, 2, 5, -1, 2, (-0.01062675516525527, 0.031191158633994506, 0.4784466137228214), (-0.01062675516525527, 0.040823159321478585, 0.4728855755330677), (-0.0, 0.8660254037844348, -0.5000000000000068), -0.01112207637950723, 0.0, 0.0, (1.0, 0.0, 0.0), 0.0, (-0.0, 0.5000000000000068, 0.8660254037844347)), (0, 2, 5, -1, 6, (0.010855557565985191, 0.0941805559064174, 0.43843431226185786), (0.010855557565985191, 0.0941805464820626, 0.4410000156541696), (-0.0, -3.673205102845747e-06, 0.9999999999932538), -0.0025657033923290225, 9.234945622780819, 3.228936004524131, (0.0, 0.9999999999932538, 3.673205102845747e-06), 0.14590856387912304, (1.0, -0.0, 0.0))), 
            'uid': 2}, 
        'block_blue': {
            'current_pos': (-0.20162748145997916, 0.09379042565818856, 0.45960969890530523), 
            'current_orn': (-0.0010699890066863072, -0.006181227283536064, 0.9240599895460498, 0.3821960011210858), 
            'current_lin_vel': (0.039142879500962915, 0.14659734410270836, -0.04759739067494311), 
            'current_ang_vel': (1.5871805226894073, -2.719713009568093, 2.214057588675123), 
            'contacts': ((0, 3, 5, -1, 6, (-0.17500839102701987, 0.09303454483742081, 0.4396849075341994), (-0.17500839102701984, 0.09303454000677441, 0.4410000114446528), (2.5328272455862733e-14, -3.6732051044613095e-06, 0.9999999999932537), -0.0013151039104622355, 7.31624108432548, 1.9810613629753318, (0.0, 0.9999999999932537, 3.6732051044613095e-06), 1.622498187244873, (1.0, 9.30359396720618e-20, -2.5328272455691866e-14)), (0, 3, 4, -1, -1, (-0.20211447621564232, 0.0649759829074997, 0.43974306250945344), (-0.19614611909444465, 0.07585949966009892, 0.4399371359793948), (0.48077242333260345, 0.8767060377375833, 0.01563330922623345), -0.012414100375862673, 0.0, 0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 3, 4, -1, -1, (-0.20193219971273177, 0.06481685461399712, 0.47969606865006825), (-0.19623919176606677, 0.07519826211056718, 0.4798811885723814), (0.48077242333260345, 0.8767060377375833, 0.01563330922623345), -0.011841377896016537, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 3, 4, -1, -1, (-0.2299823512798788, 0.09333230047201839, 0.479937614874091), (-0.22982673605397855, 0.09361607050850458, 0.47994267502485827), (0.48077242333260345, 0.8767060377375833, 0.01563330922623345), -0.0003236775204816442, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 3, 4, -1, -1, (-0.2301639984907538, 0.09349087939044448, 0.4401225426314736), (-0.2297339847067731, 0.09427502519920994, 0.44013652541861), (0.48077242333260345, 0.8767060377375833, 0.01563330922623345), -0.0008944227312372782, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641))), 
            'uid': 3}, 
        'block_pink': {
            'current_pos': (-0.2048159300643836, 0.05686644068649799, 0.45960407499986033), 
            'current_orn': (0.006367260214174141, -0.0002915188733660985, -0.2478598535296384, 0.9687749305294691), 
            'current_lin_vel': (-0.031513553573815224, -0.12285349505455018, -0.0026246457693338207), 
            'current_ang_vel': (-0.9649210821368261, -0.18659522534095357, 0.14352213903719022), 
            'contacts': ((0, 4, 5, -1, 6, (-0.17956642860895045, 0.022279538702395053, 0.4391175533707221), (-0.17956642860894878, 0.0222795317886946, 0.44099975154699544), (8.835466657234391e-13, -3.6732053678564587e-06, 0.9999999999932537), -0.0018821981762860286, 2.2834777569755933, -0.7908843972595445, (0.0, 0.9999999999932537, 3.6732053678564587e-06), -0.11510876087503996, (1.0, 3.2454483552870136e-18, -8.835466657174787e-13)), (0, 4, 3, -1, -1, (-0.19614611909444465, 0.07585949966009892, 0.4399371359793948), (-0.20211447621564232, 0.0649759829074997, 0.43974306250945344), (-0.48077242333260345, -0.8767060377375833, -0.01563330922623345), -0.012414100375862673, 0.0, 0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 4, 3, -1, -1, (-0.19623919176606677, 0.07519826211056718, 0.4798811885723814), (-0.20193219971273177, 0.06481685461399712, 0.47969606865006825), (-0.48077242333260345, -0.8767060377375833, -0.01563330922623345), -0.011841377896016537, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 4, 3, -1, -1, (-0.22982673605397855, 0.09361607050850458, 0.47994267502485827), (-0.2299823512798788, 0.09333230047201839, 0.479937614874091), (-0.48077242333260345, -0.8767060377375833, -0.01563330922623345), -0.0003236775204816442, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641)), (0, 4, 3, -1, -1, (-0.2297339847067731, 0.09427502519920994, 0.44013652541861), (-0.2301639984907538, 0.09349087939044448, 0.4401225426314736), (-0.48077242333260345, -0.8767060377375833, -0.01563330922623345), -0.0008944227312372782, 0.0, -0.0, (-0.8768131910136704, 0.48083118457981167, 0.0), 0.0, (-0.007516982594152329, -0.013707491748757206, 0.9998777923539641))), 
            'uid': 4}}, 
        'doors': {
            'base__slide': {
                'current_state': 0.0005333742072181469}, 
            'base__drawer': {
                'current_state': 0.0}}, 
            'buttons': {
                'base__button': {
                    'joint_state': 0.0, 
                    'logical_state': 0}}, 
            'switches': {
                'base__switch': {
                    'joint_state': 1.1564823173178714e-19, 
                    'logical_state': 0}}, 
            'lights': {
                'lightbulb': {
                    'logical_state': 0}, 
                    'led': {
                        'logical_state': 0}}}}
"""

def calvin_to_tapas_representation(c_obs, c_action) -> list[SceneObservation]: # type: ignore
    """
    Convert a CALVIN observation and action into a TAPAS SceneObservation.
    
    Assumptions:
      - c_obs is a dict with keys:
          "rgb_obs": dict of camera images (e.g. {"wrist": numpy array ...})
          "depth_obs": dict of depth images (e.g. {"wrist": numpy array ...})
          "robot_obs": 1D array with:
              [tcp_pos (3), tcp_orn (4), gripper_opening_width (1), arm_joint_states (7), gripper_action (1)]
          "robot_info": dict containing at least:
              "arm_joint_states": list of 7 joint positions,
              "gripper_action": a scalar.
          "scene_info": dict with a key "movable_objects": dict where each value has
              "current_pos": (3,) and "current_orn": (4,)
      - c_action is a (batch, 7)-shaped array or list convertible to a torch tensor.
      - For cameras, we assume a single camera "wrist" and use dummy intrinsics and extrinsics.
      - Joint velocities are not available in c_obs so we default them to zeros.
      - Batch size is inferred from c_action.
    
    Returns:
      A list containing one SceneObservation instance (batched).
    """
    # --- Process robot state ---
    if c_obs is None:
        return None
    robot_obs = c_obs.get('robot_obs')
    if robot_obs is None:
        return None
    
    # Extract TCP position (first 3 numbers) and TCP orientation (next 4 numbers)
    tcp_pos = np.array(robot_obs[:3])
    tcp_orn = np.array(robot_obs[3:7])  # assumed to be a quaternion (qx, qy, qz, qw)
    # Compute end-effector pose as 7 numbers: [tcp_pos, tcp_orn]
    ee_pose = np.concatenate([tcp_pos, tcp_orn])  # shape (7,)

    # Extract gripper opening width and gripper action (positions in the vector)
    # (Here we assume gripper_opening_width is at index 7 and gripper_action is the last element)
    gripper_opening_width = robot_obs[7]
    gripper_action = robot_obs[-1]
    
    # Arm joint positions from robot_info (assumed to be 7 numbers)
    arm_joint_pos = np.array(c_obs['robot_info']['arm_joint_states'])
    # If velocities are not provided, we set them to zeros
    arm_joint_vel = np.zeros_like(arm_joint_pos)

    # --- Process action ---
    # Convert c_action to a torch tensor (assume float32)
    action_tensor = torch.tensor(c_action, dtype=torch.float32)  # shape: (batch_size, 7)
    batch_size = action_tensor.shape[0]

    # --- Process camera observations ---
    # We assume that c_obs["rgb_obs"] and c_obs["depth_obs"] are dicts mapping camera names to images.
    # Here we use the "wrist" camera.
    # Convert these observations to torch tensors.
    rgb_arr = c_obs['rgb_obs'].get('wrist', None)
    depth_arr = c_obs['depth_obs'].get('wrist', None)
    rgb_tensor = torch.tensor(rgb_arr, dtype=torch.float32) if rgb_arr is not None else None
    depth_tensor = torch.tensor(depth_arr, dtype=torch.float32) if depth_arr is not None else None
    # For demonstration, we create dummy camera intrinsics (3x3 identity) and extrinsics (4x4 identity).
    dummy_intr = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    dummy_extr = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
    # Create a SingleCamObservation for the "wrist" camera.
    cam_obs = SingleCamObservation(
        rgb=rgb_tensor,
        depth=depth_tensor,
        extr=dummy_extr,
        intr=dummy_intr,
        mask=None,
        batch_size=torch.Size([batch_size])
    )
    # Specify camera order (here only "wrist")
    cam_order = CameraOrder(order=('wrist',), batch_size=torch.Size([batch_size]))
    # Build a dictionary mapping camera names to observations.
    cameras_dict = {
        'wrist': cam_obs,
        '_order': cam_order
    }
    cameras_tensordict = dict_to_tensordict(cameras_dict)

    # --- Process object poses ---
    # From scene_info["movable_objects"], extract for each object a 7-dimensional pose [pos, quat]
    movable_objects = c_obs.get('scene_info', {}).get('movable_objects', {})
    object_poses_dict = {}
    for name, obj in movable_objects.items():
        pos = np.array(obj['current_pos'])  # shape (3,)
        orn = np.array(obj['current_orn'])  # shape (4,)
        obj_pose = np.concatenate([pos, orn])  # shape (7,)
        # Convert and repeat to have shape (batch_size, 7)
        obj_pose_tensor = torch.tensor(obj_pose, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        # Use the object name as the key (or you could standardize names as obj000, obj001, etc.)
        object_poses_dict[name] = obj_pose_tensor
    object_poses_tensordict = dict_to_tensordict(object_poses_dict)

    # --- Process additional robot information ---
    # For joint positions and velocities, create tensors and repeat for batch.
    joint_pos_tensor = torch.tensor(arm_joint_pos, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    joint_vel_tensor = torch.tensor(arm_joint_vel, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    # For ee_pose, repeat it to match batch size.
    ee_pose_tensor = torch.tensor(ee_pose, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    # Gripper state (we use gripper_action here) as a tensor of shape (batch_size, 1)
    gripper_state_tensor = torch.tensor([gripper_action], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    # For feedback, we create a zero tensor.
    feedback_tensor = torch.zeros(batch_size, 1, dtype=torch.float32)

    # --- Assemble the SceneObservation ---
    scene_obs = SceneObservation(
        action=action_tensor,             # shape: (batch_size, 7)
        cameras=cameras_tensordict,         # LazyStackedTensorDict of camera observations
        ee_pose=ee_pose_tensor,             # shape: (batch_size, 7)
        feedback=feedback_tensor,           # shape: (batch_size, 1)
        gripper_state=gripper_state_tensor, # shape: (batch_size, 1)
        joint_pos=joint_pos_tensor,         # shape: (batch_size, 7)
        joint_vel=joint_vel_tensor,         # shape: (batch_size, 7)
        object_poses=object_poses_tensordict, # LazyStackedTensorDict of object poses
        kp=None,
        batch_size=torch.Size([batch_size])
    )

    # Return as a list (if required by your interface)
    return [scene_obs]


""" 
SceneObservation(
    action=Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
    cameras=LazyStackedTensorDict(
        fields={
            _order: CameraOrder(
                order=('wrist',),
                batch_size=torch.Size([83]),
                device=None,
                is_shared=False),
            wrist: SingleCamObservation(
                depth=Tensor(shape=torch.Size([83, 256, 256]), device=cpu, dtype=torch.float32, is_shared=False),
                extr=Tensor(shape=torch.Size([83, 4, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                intr=Tensor(shape=torch.Size([83, 3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                rgb=Tensor(shape=torch.Size([83, 3, 256, 256]), device=cpu, dtype=torch.float32, is_shared=False),
                descriptor=None,
                mask=None,
                batch_size=torch.Size([83]),
                device=None,
                is_shared=False)},
        exclusive_fields={
        },
        batch_size=torch.Size([83]),
        device=None,
        is_shared=False,
        stack_dim=0),
    ee_pose=Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
    feedback=Tensor(shape=torch.Size([83, 1]), device=cpu, dtype=torch.float32, is_shared=False),
    gripper_state=Tensor(shape=torch.Size([83, 1]), device=cpu, dtype=torch.float32, is_shared=False),
    joint_pos=Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
    joint_vel=Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
    object_poses=LazyStackedTensorDict(
        fields={
            obj000: Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
            obj001: Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False),
            obj002: Tensor(shape=torch.Size([83, 7]), device=cpu, dtype=torch.float32, is_shared=False)},
        exclusive_fields={
        },
        batch_size=torch.Size([83]),
        device=cpu,
        is_shared=False,
        stack_dim=0),
    kp=None,
    batch_size=torch.Size([83]),
    device=None,
    is_shared=False) 
"""

def tapas_to_calvin_representation(t_obs: list[SceneObservation]) -> dict: # type: ignore
    return {}