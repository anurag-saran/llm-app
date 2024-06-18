import base64
from collections import defaultdict
import json
import os
import pathlib

from jinja2 import Environment, FileSystemLoader


class Report:
    def __init__(self, llm_type, model, prompts_manager: "MongoPromptsManager"):
        self.llm_type = llm_type
        self.model = model
        self.prompts_manager = prompts_manager

        self.current_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
        self.results_dir = self.current_dir / "../results"
        os.makedirs(self.results_dir, exist_ok=True)

        with open(self.current_dir / "../static/images/logo.png", "rb") as image_file:
            self.base64_logo = base64.b64encode(image_file.read()).decode('utf-8')

        output_data_model = llm_type + "/" + model
        if llm_type == "openai":
            output_data_model = "OpenAI "
            if model == "gpt-3.5-turbo":
                output_data_model += "ChatGPT 3.5"
            elif model == "gpt-4":
                output_data_model += "ChatGPT 4.0"
            else:
                output_data_model += model
        elif llm_type == "huggingface":
            output_data_model = f"HuggingFace {model.capitalize()}"

        self.output_data_model = output_data_model
        self.html_data = {
            "title": "Report",
            "model": self.output_data_model,
            "base64_logo": self.base64_logo,
            "detectors": defaultdict(dict),
        }

    async def _get_data_for_report(self, detector_class):
        prompts = await self.prompts_manager.retrieve_prompts_by_detector(
            detector_class.detector_type
        )
        beautify_results = []

        for prompt in prompts:
            result = {
                "prompt_id": prompt["prompt_id"],
                "status": prompt[detector_class.detector_type].get("status", "error"),

            }
            for detector_type in ["toxicity", "relevance"]:
                if detector_type in prompt["detectors"]:
                    result["score"] = prompt[detector_class.detector_type].get("score", 0)
            if "privacy" in prompt["detectors"]:
                result["categories"] = prompt[detector_class.detector_type].get("categories", [])

            result["prompt"] = prompt["prompt"]
            result["llm_response"] = prompt["llm_response"]
            result["metadata"] = prompt["metadata"]
            beautify_results.append(result)

        return beautify_results

    async def save_result_to_json(self, detector_class):
        path_to_save = f"{self.results_dir}/result_{detector_class.detector_type}.json"
        beautify_results = await self._get_data_for_report(detector_class)
        with open(path_to_save, 'w') as file:
            json.dump(beautify_results, file, indent=4)

    async def add_report_data(self, data: dict, detector_class: "BaseDetector"):
        detector_data = await self._get_data_for_report(detector_class)
        self.html_data["detectors"][detector_class.detector_type].update(data)
        self.html_data["detectors"][detector_class.detector_type]["benchmark_data"] = detector_data

    def generate(self, debug: bool):
        templates_dir = self.current_dir / "../templates"
        template = Environment(loader=FileSystemLoader(templates_dir)).get_template("index.html")

        path_to_save = f"{self.results_dir}/report.html"

        html_output = template.render(**self.html_data, debug=debug)

        with open(path_to_save, "w") as file:
            file.write(html_output)
