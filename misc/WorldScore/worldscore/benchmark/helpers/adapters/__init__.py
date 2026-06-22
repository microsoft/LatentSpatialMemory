from worldscore.benchmark.utils.utils import layout_info


def prompt_adapter(inpainting_prompt_list, layout):
    layout_type = layout_info[layout]["layout_type"]
    if layout_type == "intra":
        return [inpainting_prompt_list[0]] * len(inpainting_prompt_list)
    elif layout_type == "inter":
        return inpainting_prompt_list
