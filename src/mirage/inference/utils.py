"""
Inference Utils.
"""

MAX_FRAMES_PER_ITERATION = 33  # Wan model maximum: 4*8+1 = 33 frames


def validate_num_frames(num_frames: int) -> bool:
    """
    Validate that num_frames follows the required format: 4N+1.

    For I2V model (Wan):
    - Total frames must be 4N+1 format (5, 9, 13, ..., 33, 37, 41, ...)
    - First iteration: model generates 33 frames, frame 0 is input condition, frames 1-32 are new
    - Subsequent iterations: model generates 33 frames, frame 0 overlaps with previous last frame,
      so each iteration adds 32 NEW frames

    Valid total frames: 33, 65, 97, 129, ... = 33 + 32*k = 1 + 32*(k+1)
    Or any 4N+1 up to 33 for single iteration: 5, 9, 13, 17, 21, 25, 29, 33

    Args:
        num_frames: Total number of frames to generate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If num_frames doesn't follow the required format
    """
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")

    # num_frames 必须是 4N+1 格式
    if (num_frames - 1) % 4 != 0:
        valid_examples = [4 * n + 1 for n in range(1, 12)]  # [5, 9, 13, ..., 45]
        raise ValueError(
            f"num_frames ({num_frames}) must be 4N+1 format.\n"
            f"Valid examples: {valid_examples}, ..."
        )

    return True


def compute_iteration_plan(num_frames: int) -> list:
    """
    Compute the iteration plan for generating num_frames.

    For I2V model:
    - First iteration generates up to 33 frames (model output), all are "new" in final output
    - Each subsequent iteration generates 33 frames from model, but frame 0 overlaps
      with previous iteration's last frame, so only 32 frames are NEW

    Total frames = 33 + 32*(num_iterations-1) for multi-iteration
    Or any 4N+1 <= 33 for single iteration

    Args:
        num_frames: Total number of frames to generate (must be 4N+1)

    Returns:
        list: List of tuples (output_start, output_end, model_frames)
              - output_start: First frame index in final output video
              - output_end: Last frame index + 1 in final output video
              - model_frames: Number of frames the model generates this iteration (4N+1)

    Example:
        num_frames=33 -> [(0, 33, 33)]  # 1 iteration
        num_frames=65 -> [(0, 33, 33), (33, 65, 33)]  # 2 iterations, 2nd adds 32 new
        num_frames=97 -> [(0, 33, 33), (33, 65, 33), (65, 97, 33)]  # 3 iterations
        num_frames=17 -> [(0, 17, 17)]  # Single short iteration
    """
    validate_num_frames(num_frames)

    plan = []

    if num_frames <= MAX_FRAMES_PER_ITERATION:
        # Single iteration case
        plan.append((0, num_frames, num_frames))
    else:
        # Multi-iteration case
        # First iteration: 33 frames
        plan.append((0, MAX_FRAMES_PER_ITERATION, MAX_FRAMES_PER_ITERATION))
        current_output_frame = MAX_FRAMES_PER_ITERATION

        # Subsequent iterations: each adds 32 new frames (model generates 33, but frame 0 overlaps)
        remaining = num_frames - MAX_FRAMES_PER_ITERATION
        while remaining > 0:
            # Model always generates 33 frames (or less if remaining + 1 < 33)
            # But we only count 32 as "new" (frame 0 is overlap)
            new_frames_this_iter = min(32, remaining)
            model_frames = new_frames_this_iter + 1  # +1 for the overlapping frame 0

            # Ensure model_frames is 4N+1
            if (model_frames - 1) % 4 != 0:
                # Round up to next 4N+1
                model_frames = ((model_frames - 1) // 4 + 1) * 4 + 1

            output_start = current_output_frame
            output_end = current_output_frame + new_frames_this_iter

            plan.append((output_start, output_end, model_frames))

            current_output_frame = output_end
            remaining -= new_frames_this_iter

    return plan
