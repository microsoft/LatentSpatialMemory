from worldscore.benchmark.utils.utils import center_crop, generate_prompt_from_list


def adapter_4dfy(config, data, helper):
    output_dir, image_path, inpainting_prompt_list, camera_path = (
        data["output_dir"],
        data["image_path"],
        data["inpainting_prompt_list"],
        data["camera_path"],
    )

    inpainting_prompt = generate_prompt_from_list(
        inpainting_prompt_list,
        camera_path,
        static=True if config["visual_movement"] == "static" else False,
    )

    helper.prepare_data(output_dir, data)
    image_path = center_crop(image_path, config["resolution"], output_dir)
    return inpainting_prompt
