"""Task semantics and robot-neutral interface selection; no answer coordinates."""
import os
PEG='Pick up the loose bolt, insert its shaft through the upper-right bolt hole in the raised hub-cover annulus (about two o-clock from its large center bore in the fixed task-camera view) into the corresponding casing opening, seat the head on the cover, release, and withdraw the gripper. Seating means the underside of the head rests on the cover; the head remains visible above the surface. Do not force the top of the head flush with or below the cover. Complete the assembly task. Call observe first.'
HUB_INSTRUCTIONS={
    'hub-cover':'Pick up the loose output hub cover, identify its matching output opening in the casing top from current public visual geometry, fit and physically seat it, release it, and withdraw. Do not use scene coordinates or historical fit poses, or assume the nearest opening is the target. Complete the assembly task. Call observe first.',
    'hub-cover-input':'Pick up the loose input hub cover, identify its matching input opening in the casing top by comparing current public part/opening geometry, fit and physically seat it, release it, and withdraw. Do not use scene coordinates or historical fit poses, or assume the nearest opening is the target. Call observe first.',
    'hub-cover-small':'Pick up the loose small hub cover, identify the matching small opening in the casing top by comparing current public part/opening geometry, fit and physically seat it, release it, and withdraw. Do not use scene coordinates or historical fit poses, or assume the nearest opening is the target. Call observe first.',
}
HUB_TASKS=set(HUB_INSTRUCTIONS)
GEAR_INSTRUCTIONS={
    'output-gear-onto-shaft':'Pick up the loose output gear, identify the matching exposed output shaft from the current public observation, align the gear bore with the shaft axis, slide it along the shaft until the physical shoulder/seat is reached, release, and withdraw. Use observed geometry and public CAD features; do not use scene coordinates or historical fit poses as task answers. Re-observe to check retention and seating before finishing. Call observe first.',
    'transfer-gear-onto-shaft':'Pick up the loose transfer gear, identify the matching exposed transfer shaft from the current public observation, align the gear bore with the shaft axis, slide it along the shaft until the physical shoulder/seat is reached, release, and withdraw. Use observed geometry and public CAD features; do not use scene coordinates or historical fit poses as task answers. Re-observe to check retention and seating before finishing. Call observe first.',
}
GEAR_TASKS=set(GEAR_INSTRUCTIONS)
SHAFT_INSTRUCTIONS={
    'output-shaft-into-base':'Pick up the loose output shaft assembly, identify its matching bearing bore in the casing base from current public observation, align the shaft axis with that bore, insert until the physical shoulder/seat is reached, release and withdraw. Use observed geometry and public CAD features; do not use historical reset poses as target answers. Re-observe seating and stability before finishing. Call observe first.',
    'transfer-shaft-into-base':'Pick up the loose transfer shaft assembly, identify its matching bearing bore in the casing base from current public observation, align the shaft axis with that bore, insert until the physical shoulder/seat is reached, release and withdraw. Use observed geometry and public CAD features; do not use historical reset poses as target answers. Re-observe seating and stability before finishing. Call observe first.',
    'input-shaft-into-base':'Pick up the loose input shaft assembly, identify its matching bearing bore in the casing base from current public observation, align the shaft axis with that bore, insert until the physical shoulder/seat is reached, release and withdraw. Use observed geometry and public CAD features; do not use historical reset poses as target answers. Re-observe seating and stability before finishing. Call observe first.',
}
SHAFT_TASKS=set(SHAFT_INSTRUCTIONS)

def shaft_compound_specification(task):
    """Public internal CAD layout only; no casing socket, reset, or target pose.

    Offsets are from magic_assembly.HARDCODED_FIT_OFFSETS and the seven standalone
    shaft_subassembly_steps. Bearing plugs are missing in public CAD: their fit
    is a provisional CAD-root placement, not a verified attachment/physical seat.
    """
    layouts = {
        'output-shaft-into-base': ('Output', [
            ('Output_Gear', 'sub_step_output_gear', [0,15.5,0], [-90,0,0]),
            ('Bearing_Output_Top', 'sub_step_bearing_output_top', [0,51.156502,0], [0,0,-90]),
            ('Bearing_Output_Bottom', 'sub_step_bearing_output_bottom', [0,-51.156502,0], [0,0,90]),
        ]),
        'transfer-shaft-into-base': ('Transfer', [
            ('Transfer_Gear', 'sub_step_transfer_gear', [22.1,0,0], [0,90,0]),
            ('Bearing_Transfer', 'sub_step_bearing_transfer', [-50.563,0,0], [0,0,0]),
        ]),
        'input-shaft-into-base': ('Input', [
            ('Bearing_Input_Bottom', 'sub_step_bearing_input_bottom', [0,-100.75,0], [0,0,90]),
            ('Bearing_Input_Top', 'sub_step_bearing_input_top', [0,2.732481,0], [0,0,-90]),
        ]),
    }
    if task not in layouts:
        return None
    kind, members = layouts[task]
    return {
        'shaft_root': kind + '_Shaft',
        'cad_file': kind + ' Shaft Assembly.usd',
        'cache_file': 'shaft_' + kind.lower() + '_compound.npz',
        'placement_units': 'source CAD coordinates; scale once to metres',
        'source': 'assembly/shaft_subassembly_steps.yaml; runtime/magic_assembly.py HARDCODED_FIT_OFFSETS',
        'placement_status': 'provisional_source_fits_missing_bearing_plugs; physical calibration pending',
        'members': [{'name': name, 'source_step': step, 'translate': t, 'rotate_xyz': r}
                    for name, step, t, r in members],
        'input_shaft_open_mesh_provisional': kind == 'Input',
    }


def shaft_compound_paths(task, run_dir):
    """Shared backend/actor paths for the effective per-episode public CAD."""
    from pathlib import Path
    spec = shaft_compound_specification(task)
    directory = Path(run_dir).resolve() / 'checks/public_cad_models'
    return directory / spec['cad_file'], directory / spec['cache_file']

OTHER_INSTRUCTIONS={
    'm10-casing-bolt':'Pick up one loose M10 casing bolt. From the current public observation, identify an open bolt bore on the casing top, align the shank to that bore, insert it until the head seats, release, and withdraw. Do not use historical socket coordinates. Verify physical seating before finishing. Call observe first.',
    'm10-casing-nut':'Pick up one loose M10 casing nut and install it onto the exposed M10 casing bolt in the fixture. Align the thread axes, engage the thread, and rotate while advancing only when the observed geometry supports engagement. Do not treat pushing or placing the nut as threaded completion. Release and verify retention. Call observe first.',
    'casing-top-to-base':'Pick up the casing top and mate it with the casing base. Use current public geometry to align the mating flanges and all visible shaft clearances; lower under controlled motion until the flanges physically seat, then release and verify stability. Do not use historical fit coordinates. Call observe first.',
    'oil-level-indicator':'Pick up one oil-level indicator. Identify its corresponding opening on the casing base from current public geometry, align its insertion axis, insert until physically seated, release, and verify retention. Do not use historical reset coordinates. Call observe first.',
    'breather-plug':'Pick up the breather plug. Identify the side-facing breather port on the casing base from current public geometry; distinguish it from the large shaft bore and the smaller top-face holes. Align the plug axis with that port, insert until physically seated, release, and verify retention. After lifting, check finger contact and object visibility. If the plug disappears while the arm may occlude its last location, safely retract the arm to clear the view and observe again before concluding it is lost or asking for help; if visible and not retained, replan the grasp. Do not use historical reset coordinates. Call observe first.',
    'bearing-32013-onto-output-shaft':'Pick up the 32013 bearing assembly and fit it over the matching output shaft seat. Identify the shaft and bearing axes from current public geometry, align them, slide the bearing to its physical shoulder, release, and verify retention. Do not use historical fit coordinates. Call observe first.',
    'bearing-transfer-onto-shaft':'Pick up the 32205 bearing assembly and fit it over the matching transfer shaft seat. Identify the shaft and bearing axes from current public geometry, align them, slide the bearing to its physical shoulder, release, and verify retention. Do not use historical fit coordinates. Call observe first.',
    'bearing-input-bottom-onto-shaft':'Pick up the 30304 bearing assembly and fit it over the matching input shaft seat. Identify the shaft and bearing axes from current public geometry, align them, slide the bearing to its physical shoulder, release, and verify retention. Do not use historical fit coordinates. Call observe first.',
    'bearing-input-top-onto-shaft':'Pick up the 32205 bearing assembly and fit it over the input shaft onto the middle shoulder, not the first shoulder near the exposed shaft end. Passing over or resting at the shaft-end shoulder is not completion: the intended seat is farther along the shaft toward its middle. Identify the shaft axis, intermediate steps and intended middle shoulder from current public geometry and public CAD; align the bearing bore, slide it to that middle seat, release, and verify retention and seating. This is a distinct input-shaft seat, not the transfer-shaft bearing task. Do not use historical fit coordinates or force the bearing past an incompatible step. Call observe first.',
}
OTHER_TASKS=set(OTHER_INSTRUCTIONS)
SUPPORTED_TASKS={'peg'}|HUB_TASKS|GEAR_TASKS|SHAFT_TASKS|OTHER_TASKS
GENERIC_SYSTEM='''You control a physical robot in Isaac through RPent tools. Complete the task using current public RGB-D and measured robot state. Tasks are not assumed physically feasible. Use decision(continue), ask_human_help or stop as supported by evidence; finish(success) requires independent native success. The main RGB camera is fixed; the second image is the same frame with a coordinate grid and unlabelled image ellipse IDs, not semantic object labels or a wrist camera. back_project measures a visible depth surface, not semantic identity or the true object center. Coordinates are world metres and quaternions wxyz. Robot arm/TCP and jaw information comes from the public capability profile. select_arm chooses the active arm. Ordinary move and gripper tools perform physical control, not object attachment. R1 exposes move_both and independent grippers; choose single- or dual-arm execution yourself. Use plan_grasp(pixel_xy,bbox_xyxy) to generate full 6D grasp candidates from a visible object or grippable rim region, screened by measured robot-model IK, jaw span and visible-depth clearance. Candidate screening cannot guarantee clearance from hidden surfaces. Candidate estimated_grasp_pixel_xy and contact-pair pixels are projections into the same raw image, not semantic labels; use them and observed_component_image_bbox_xyxy to check that a rim grasp belongs to your selected object. World bounds are in metres; compare with the public CAD extent rather than guessing the object from world coordinates. execute_grasp(candidate_id) executes the selected pose; omit grip_force_n initially to use the public-mass-based load estimate. Override pressure gradually only when current contact/retention evidence calls for it; historical memory pressure values are not calibrated to this scene. inspect actual finger forces, width, movement and images afterwards. It does not certify retention. measure_feature estimates a local geometric feature at your selected pixel. assemble transports the measured moving feature toward its matching target feature and performs bounded contact search; the offset is a geometric contact-face offset derived from public CAD or visible dimensions, not a target world coordinate. Peg, hub-cover, and gear-to-shaft variants share these tools and physical skill code; target identities and feature choices are yours. The public geometric_patterns catalog lists all detected circular-pattern hypotheses from current RGB-D with relative image clock hours. It does not name the task target; select the correct centre/raised region from the raw image and inspect the listed candidates. measure_pattern reports geometric clock hours of all detected candidate contours around your selected centre: 12 up and 3 right. It does not select task targets. execute_grasp may return a public-CAD-derived held datum and two held_axis_features for elongated visible parts. A CAD datum is not automatically a physical seating face: inspect feature_provenance and public CAD dimensions; use seat_offset=0 only when that relation is supported by geometry. Endpoint features use measured-robot transport of pregrasp points. The CAD feature additionally attempts current public RGB-D registration after lift; inspect inhand_registration status. Subsequent transport still assumes a rigid grasp and is not GT. Inspect the image and choose the intended seating end. A held CAD feature may report a canonical required_release_clearance_m estimate. assemble recomputes conservative whole-hand clearance at the actual aligned pose against the outward target plane; opposite axis signs alone do not justify flipping a seating face. Default axis_alignment=nearest preserves the closest coaxial orientation, while parallel or antiparallel must be chosen explicitly if directed-axis evidence supports them. After checking alignment, explicitly open the gripper, retract and settle; physical gravity seating is not guaranteed. assemble can reorient a flat part by aligning its tracked axis to the target normal; do not confuse the tool approach axis with the object axis. A flat long part may need reorientation: use the observed component axis and current image, not an assumption that the bolt is upright. Do not infer success from a tool completion status. Inspect retention using current images and robot/tactile feedback. Finger forces are unfiltered total contacts and do not identify objects. Use direct moves to align, insert, release and retract. No task-specific target coordinates or scripted insertion trajectory are supplied. Respect actual motion timeouts and emergency stops. Never read simulator object poses, private evaluator state or credentials. Human help is recorded, not automatically executed.'''
GENERIC_SYSTEM+=''' Before approaching a grasp, compare the observed part/contact geometry with the public jaw range and complete finger geometry. Distinguish outside diameter, bore diameter and local wall thickness; a circular feature is not automatically the outer boundary. For internal expansion, width_m is only the inner gap: both finger bodies must fit inside the bore before opening. Use the declared TCP-to-pad geometry rather than guessing fingertip height. Read plan_grasp screening counts to distinguish generation, contact-band, finger-body collision and IK failures; changing only the ROI cannot necessarily fix a finger-body collision. Manual moves remain available but do not inherit plan_grasp screening. Reobserve after approach and reconcile the actual TCP, opening and contact direction before closure. Total finger force can result from pressing on a top face; compare closing-axis pressure and force vectors, then verify target co-motion in a small lift. Approaching under a fixed camera does not create a closer camera view. These are general robot/tool semantics, not a task-specific grasp prescription.'''
GENERIC_SYSTEM+=''' plan_grasp and execute_grasp currently implement compressive closing grasps, not internal expansion. A large OD excludes a whole-diameter enclosing grasp but does not by itself exclude a local inner/outer wall pinch. Do not infer that internal expansion is the only alternative. Changing an ROI cannot add an unsupported grasp mode.'''
GENERIC_SYSTEM+=''' For a manual grasp, tool_orientation converts your chosen approach and closing directions to a unit quaternion. inspect_pose reports robot IK and estimated Panda hand/table clearance for your proposed TCP pose and jaw opening without moving. Check hand clearance, not just fingertip height, especially for horizontal approaches. A pose check is not a transit-path certificate. Reobserve during staged approach rather than assuming one long coupled translation/rotation will reach safely.'''
GENERIC_SYSTEM+=''' view_public_cad shows only the task-named moving-part reference CAD in local-axis views, not its scene location. Use shape comparison and component_diagnostics.registration.fit_supported when identity is uncertain; do not treat an IK check as confirmation of object identity or a grasp. Reference orientation may differ from the live scene.'''
def generic():return True
def instruction():
    task=os.environ.get('ASSEMBLY_TASK','peg')
    if task=='peg':return PEG
    if task in HUB_INSTRUCTIONS:return HUB_INSTRUCTIONS[task]
    if task in GEAR_INSTRUCTIONS:return GEAR_INSTRUCTIONS[task]
    if task in SHAFT_INSTRUCTIONS:return SHAFT_INSTRUCTIONS[task]
    if task in OTHER_INSTRUCTIONS:
        prompt=OTHER_INSTRUCTIONS[task]
        if task=='casing-top-to-base':
            if os.environ.get('ASSEMBLY_ROBOT','panda')=='r1':
                prompt += ' The available R1 has two arms; use one or both arms based on observed grasp and load stability. If it cannot be safely handled, request specific human assistance.'
            else:
                prompt += ' Only the Panda single arm is available; do not assume a second arm or call dual-arm tools. If the load cannot be safely handled with the available arm, request specific human assistance.'
        return prompt
    raise ValueError(f'No task prompt is registered for {task!r}')

def skill_specification():
    import json
    from pathlib import Path
    task=os.environ.get('ASSEMBLY_TASK','peg')
    try:profile=json.loads((Path(__file__).resolve().parent/'config/assembly_skills.json').read_text())[task]
    except KeyError as exc:raise ValueError(f'No assembly skill is registered for {task!r}') from exc
    compound = shaft_compound_specification(task)
    if compound:
        profile['initial_compound'] = compound
        # Without an episode, retain the root CAD reference for offline callers.
        # Every running episode uses the complete generated assembly.
        if os.environ.get('ASSEMBLY_RUN_DIR'):
            asset, cache = shaft_compound_paths(task, os.environ['ASSEMBLY_RUN_DIR'])
            profile.update(moving_part=compound['shaft_root'] + ' initial compound assembly',
                           moving_cad_file=compound['cad_file'], moving_asset_path=str(asset),
                           moving_cad_cache=str(cache))
    return profile

def public_skill_specification():
    """Expose actor-relevant public CAD facts without evaluator targets/calibration."""
    profile=skill_specification()
    public_keys={
        'task_family','moving_part','mating_feature','moving_cad_file','moving_cad_cache',
        'moving_feature_local_m','moving_feature_axis_local','feature_provenance',
        'relation','contact_mode','verification','initial_compound',
    }
    return {key:value for key,value in profile.items() if key in public_keys}
