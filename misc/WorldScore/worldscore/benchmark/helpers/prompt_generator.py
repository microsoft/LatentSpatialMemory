import base64
import io
import json
import os
import time
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from worldscore.benchmark.utils.utils import get_scene_num

# Load the .secrets file
try:
    load_dotenv(".secrets")
except Exception as e:
    print(f"Error loading .secrets file: {e}")
    print(
        "Please make sure you have a .secrets file in the root directory of the project."
    )

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


class TextPromptGen(object):
    def __init__(self):
        super(TextPromptGen, self).__init__()
        self.model = "gpt-4o"
        self.scene_num = 0
        self.id = 0
        self.base_content = (
            "Please generate next scene based on the given scene/scenes information:"
        )
        self.content = self.base_content

    def wonder_next_scene(
        self, style=None, entities=None, scene_name=None, prompt=None
    ):
        ######################################
        # Input ------------------------------
        # scene_name: str
        # entities: List(str) ['entity_1', 'entity_2', 'entity_3']
        # style: str
        # prompt: str
        ######################################
        # Output -----------------------------
        # output: dict {'scene_name': [''], 'entities': ['', '', '']}
        ######################################

        self.scene_num += 1
        self.id += 1
        if prompt is not None:
            scene_content = (
                "\nScene "
                + str(self.id)
                + ": "
                + str(prompt)
                + "; Style: "
                + str(style)
            )
        else:
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            scene_content = (
                "\nScene "
                + str(self.id)
                + ": "
                + "{Scene name: "
                + str(scene_name).strip(".")
                + "; Entities: "
                + str(entities)
                + "; Style: "
                + str(style)
                + "}"
            )
        self.content += scene_content

        messages = [
            {
                "role": "system",
                "content": "Imagine you are moving through a series of interconnected scenes. Each scene features 1 or 2 key entities that adapt to the context of the scene and naturally transition to the next. Generate the name of the next sequential scene along with its 1 or 2 most significant entities. Ensure that the entities seamlessly fit with and evolve from the previous scene, without exceeding the limit of two entities. Use this JSON format for your response:\n \
                        {'scene_name': ['scene_name'], 'entities': ['entity_1', 'entity_2']}",
            },
            {"role": "user", "content": self.content},
        ]

        for i in range(10):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    # timeout=5,
                )
                response = response.choices[0].message.content
                try:
                    print(response)
                    output = eval(response)
                    _, _ = output["scene_name"], output["entities"]
                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output["scene_name"], str):
                        output["scene_name"] = [output["scene_name"]]
                    if isinstance(output["entities"], str):
                        output["entities"] = [output["entities"]]
                    assert len(output["entities"]) <= 2
                    break
                except Exception as e:
                    assistant_message = {"role": "assistant", "content": response}
                    user_message = {
                        "role": "user",
                        "content": "Something went wrong. The output is not json format, or the output is not exactly the format I want. please try again:\n"
                        + self.content,
                    }
                    messages.append(assistant_message)
                    messages.append(user_message)
                    print(
                        "An error occurred when transfering the output of chatGPT into a dict, chatGPT4, let's try again!",
                        str(e),
                    )
                    continue
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print("Wait for a second and ask chatGPT4 again!")
                time.sleep(1)
                continue

        return output


class PromptGen(object):
    def __init__(self, config):
        super(PromptGen, self).__init__()
        self.config = config
        self.model = "gpt-4o"
        self.save_prompt = True
        self.root_path = config["benchmark_root"]
        self.text_prompt_generator = TextPromptGen()
        self.prompts = []

    def clear(self):
        self.text_prompt_generator.scene_num = 0
        self.text_prompt_generator.id = 0
        self.text_prompt_generator.content = self.text_prompt_generator.base_content
        self.prompts = []

    def read_json(self):
        file_path = os.path.join(
            self.config["dataset_root"],
            self.config["visual_movement"],
            f"{self.config['visual_style']}_selected.json",
        )
        try:
            with open(file_path) as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
        return data

    def write_all_content(self, save_dir=None):
        if save_dir is None:
            save_dir = Path(self.root_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "generation_log.txt", "w") as f:
            f.write(self.text_prompt_generator.content)
        return

    def write_json(self, output, name, save_dir=None):
        if save_dir is None:
            save_dir = (
                Path(self.root_path)
                / f"data/images/{self.config['scene_type']}/{self.config['visual_style']}"
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_dir / f"{name}.json", "w") as json_file:
                json.dump(output, json_file, indent=4)
        except Exception as e:
            pass
        return

    def generate_prompt(self, prompt):
        if isinstance(prompt, dict):
            scene_name = prompt["scene_name"]
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            entities = prompt["entities"]
            prompt_text = scene_name + ", "
            for i, entity in enumerate(entities):
                if i == 0:
                    prompt_text += entity
                else:
                    prompt_text += ", " + entity
        else:
            prompt_text = prompt

        print("PROMPT TEXT: ", prompt_text)

        return prompt_text

    def generate_prompts(self, regenerate=False):
        datas = self.read_json()

        for i, data_item in tqdm(enumerate(datas)):
            if "prompt_list" in data_item and not regenerate:
                print(f"({i}) Prompt list already exists: {data_item['prompt_list']}")
                continue
            layout = data_item["camera_path"]
            scene_num = get_scene_num(layout)

            scene_style = data_item["style"]
            prompt = data_item["prompt"].split(",")
            scene_name = prompt[0]
            entities = prompt[1:]
            scene_dict = {"scene_name": scene_name, "entities": entities}

            self.prompts.append(data_item["prompt"])
            for i in range(scene_num):
                scene_dict = self.text_prompt_generator.wonder_next_scene(
                    scene_name=scene_name, entities=entities, style=scene_style
                )
                scene_name, entities = scene_dict["scene_name"], scene_dict["entities"]
                self.prompts.append(self.generate_prompt(scene_dict))

            assert len(self.prompts) == scene_num + 1
            data_item["prompt_list"] = self.prompts
            self.clear()

            if i % 10 == 0:
                with open(
                    os.path.join(
                        self.config["dataset_root"],
                        self.config["visual_movement"],
                        f"{self.config['visual_style']}_selected.json",
                    ),
                    "w",
                ) as f:
                    json.dump(datas, f, indent=4)
        with open(
            os.path.join(
                self.config["dataset_root"],
                self.config["visual_movement"],
                f"{self.config['visual_style']}_selected.json",
            ),
            "w",
        ) as f:
            json.dump(datas, f, indent=4)
        return
