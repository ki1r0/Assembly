"""Bounded render-only retry for Isaac's first-frame overscan metadata race."""
import traceback


def configure_full_frame_window(settings):
    """Our calibrated camera uses the entire image, without crop or overscan."""
    for index,value in enumerate((0.,0.,1.,1.)):
        settings.set_float(f'/rtx/dataWindowNDC/{index}',value)


def update_ready_camera(sim, camera, dt):
    for attempt in range(8):
        try:
            camera.update(dt if attempt == 0 else 0, force_recompute=True)
            return
        except (TypeError, RuntimeError, IndexError) as error:
            overscan_race = any(
                frame.name == '_resize_data_for_overscan' and frame.filename.endswith('annotator_utils.py')
                for frame in traceback.extract_tb(error.__traceback__)
            ) and "'NoneType' and 'NoneType'" in str(error)
            empty_annotator = isinstance(error, RuntimeError) and 'invalid for input of size 0' in str(error) and any(
                frame.name == '_process_annotator_output' and frame.filename.endswith('/sensors/camera/camera.py')
                for frame in traceback.extract_tb(error.__traceback__)
            )
            empty_overscan=False;overscan_shape=None
            cursor=error.__traceback__
            while cursor is not None:
                frame=cursor.tb_frame
                if isinstance(error,IndexError) and frame.f_code.co_name=='_resize_data_for_overscan' and frame.f_code.co_filename.endswith('annotator_utils.py'):
                    shape=getattr(frame.f_locals.get('data'),'shape',())
                    overscan_shape=tuple(shape)
                    empty_overscan=len(shape)<2 and 0 in shape
                cursor=cursor.tb_next
            if not (overscan_race or empty_annotator or empty_overscan) or attempt == 7:
                if overscan_shape is not None:error.add_note(f'Isaac overscan buffer shape: {overscan_shape}')
                raise
            sim.render()
