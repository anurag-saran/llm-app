import asyncio

import asyncclick as click
from tqdm.asyncio import tqdm as atqdm

from constants import Colors
from core import logger, settings, setup_logging
from detectors import (
    BaseDetector,
    ToxicityDetector,
    StereotypeDetector,
    FairnessDetector,
    RelevanceDetector,
    HallucinationsDetector,
    PrivacyDetector
)
from llm_clients import BaseLlmClient, llm_factory
from mongo_db_services import MongoClient, MongoPromptsManager, load_initial_data
from reports.reports import Report


db = MongoClient()
prompt_manager = MongoPromptsManager(db)
queue = asyncio.Queue()

detectors_class_map = {
    ToxicityDetector.detector_type: ToxicityDetector,
    StereotypeDetector.detector_type: StereotypeDetector,
    FairnessDetector.detector_type: FairnessDetector,
    RelevanceDetector.detector_type: RelevanceDetector,
    HallucinationsDetector.detector_type: HallucinationsDetector,
    PrivacyDetector.detector_type: PrivacyDetector
}


def _calculate_percentage(count: int, total: int) -> float:
    return (count / total * 100) if total > 0 else 0


def _clean_newlines(data: str | list) -> str | list:
    if isinstance(data, str):
        return data.replace("\n", " ")
    elif isinstance(data, list):
        return [item.replace("\n", " ") for item in data if isinstance(item, str)]
    return data


class Pipeline:
    def __init__(
            self,
            llm_client: BaseLlmClient = BaseLlmClient,
            detector: BaseDetector = BaseDetector,
    ):
        self.llm_client = llm_client
        self.detector = detector

    @staticmethod
    def _get_max_tokens(detectors: list) -> int | None:
        if HallucinationsDetector.detector_type in detectors:
            return settings.HALLUCINATION_PRIVACY_MAX_TOKENS
        if PrivacyDetector.detector_type in detectors:
            return settings.HALLUCINATION_PRIVACY_MAX_TOKENS
        if StereotypeDetector.detector_type in detectors:
            return settings.STEREOTYPE_FAIRNESS_MAX_TOKENS
        if FairnessDetector.detector_type in detectors:
            return settings.STEREOTYPE_FAIRNESS_MAX_TOKENS

        return None

    async def _execute_llm_inference(self, prompt: str, use_fairness_tokens: int | None) -> str:
        tokens = use_fairness_tokens if use_fairness_tokens else None
        return await self.llm_client.inference(prompt, max_tokens=tokens)

    async def run_llm(self, prompt: dict, pbar_llm: atqdm, concurrently: bool = False):
        if prompt.get("llm_response"):
            logger.debug(f"LLM response: {prompt['llm_response']} already processed for prompt: {prompt['prompt']}")
            return

        use_max_tokens = self._get_max_tokens(prompt["detectors"])

        if isinstance(prompt["prompt"], str):
            llm_response = await self._execute_llm_inference(prompt["prompt"], use_max_tokens)
            pbar_llm.update(1)
        elif isinstance(prompt["prompt"], list):
            if concurrently:
                tasks_for_llm = [self._execute_llm_inference(prmpt, use_max_tokens) for prmpt in prompt["prompt"]]
                llm_response = await asyncio.gather(*tasks_for_llm)
                pbar_llm.update(1)
            else:
                llm_response = []
                pbar_sub_llm = atqdm(
                    total=len(prompt["prompt"]),
                    desc=f"{Colors.blue.value}Processing sub LLM request"
                )

                for prmpt in prompt["prompt"]:
                    response = await self._execute_llm_inference(prmpt, use_max_tokens)
                    llm_response.append(response)
                    pbar_sub_llm.update(1)
                pbar_sub_llm.close()
                pbar_llm.update(1)
        else:
            logger.error(f"Invalid prompt['prompt'] type: {prompt}")
            return

        await prompt_manager.update_prompt(
            prompt_id=prompt["_id"], update_set={"llm_response": _clean_newlines(llm_response)}
        )

    async def run_detectors(self, prompt: dict, pbar_detector: atqdm, verbose: bool):

        llm_response = prompt.get("llm_response")
        if not llm_response:
            logger.info(f"Prompt {prompt['prompt']} does not have an LLM response.")
            return

        if prompt.get(self.detector.detector_type):
            logger.debug(
                f"Detector {self.detector.detector_type} already processed for prompt: "
                f"{prompt['prompt']} --> {prompt['llm_response']}"
            )
            return

        def write_to_progress_bar(llm_response: str, prompt: dict, result: dict, pbar=None):
            pbar = pbar if pbar else pbar_detector
            if not verbose:
                pbar.update(1)
                return

            color = Colors.bright_green.value if result["status"] == "passed" else (
                Colors.bright_yellow.value if result["status"] == "skipped" else Colors.bright_red.value)
            cleaned_answer = _clean_newlines(llm_response)
            pbar.write(
                f"{color}{self.detector.detector_type} test {result['status']}\n"
                f"      Prompt - {prompt['prompt']} \n"
                f"      Metadata - {prompt['metadata']} \n"
                f"      LLM answer - {cleaned_answer}\n\n{Colors.reset.value}")
            pbar.update(1)

        result = await self.detector().evaluate(
            llm_prompt=prompt["prompt"],
            llm_response=llm_response,
            metadata=prompt["metadata"]
        )
        logger.debug(f"{self.detector.detector_type} result: {result}")

        write_to_progress_bar(llm_response, prompt, result)

        await prompt_manager.update_prompt(
            prompt_id=prompt["_id"],
            update_set={self.detector.detector_type: result},
        )


@click.command()
@click.option("--llm-type", help="LLM type")
@click.option("--model", help="Model name")
@click.option("--init-db", is_flag=True, help="Initial database")
@click.option("--generate-output-file", is_flag=True, help="Generate output file")
@click.option("--iterations", help="How many times to repeat prompts")
@click.option("--concurrently", is_flag=True, help="Concurrently requests to LLM")
@click.option("--detector", help="Processed tests only for this detector")
@click.option("--debug", is_flag=True, help="on/off debug mode")
@click.option("--verbose", is_flag=True, help="on/off writing to pbar")
async def main(llm_type, model, init_db, generate_output_file, iterations, concurrently, detector, debug, verbose):
    async def worker():
        while True:
            task = await queue.get()
            try:
                await task["function"](*task["args"], **task["kwargs"])
            finally:
                queue.task_done()

    setup_logging(debug)

    if init_db:
        await load_initial_data(db)
        if not (llm_type and model):
            return

    if not (llm_type and model):
        logger.error("No LLM provided")
        return

    try:
        llm_client = llm_factory(llm_type=llm_type, model_name=model)
    except NotImplementedError as e:
        logger.exception(e)
        return

    if iterations:
        await prompt_manager.duplicate_prompts(iterations=int(iterations))

    output_data = {}
    if generate_output_file:
        report = Report(llm_type, model, prompt_manager)

    if detector:
        if detector not in detectors_class_map:
            logger.error(f"Detector - {detector} not supported")
            return
        detectors_classes = [detectors_class_map[detector]]
    else:
        detectors_classes = list(detectors_class_map.values())

    # Detectors health check
    for detector in detectors_classes:
        detector_health = await detector.health_check()
        if detector_health["health"] != "ok":
            logger.error(f"Detector - {detector.detector_type} not healthy. Status: {detector_health}")
            return
        else:
            logger.debug(f"Detector - {detector.detector_type} healthy. Status: {detector_health}")

    # Set up workers
    num_workers = settings.LLM_CONCURRENTLY_LIMIT_MAP[model] if concurrently else 1
    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    for detector in detectors_classes:
        #  LLM
        prompts = await prompt_manager.retrieve_all_prompts_with_preparations(detector.detector_type)
        total_prompts = len(prompts)
        logger.info(f"Start processing LLM request for {detector.detector_type} detector.")
        pbar_llm = atqdm(total=total_prompts, desc=f"{Colors.bright_blue.value}Processing LLM request")
        llm_pipeline = Pipeline(llm_client=llm_client)

        for prompt in prompts:
            await queue.put(
                {
                    "function": llm_pipeline.run_llm,
                    "args": [],
                    "kwargs": {"prompt": prompt, "pbar_llm": pbar_llm, "concurrently": concurrently}
                }
            )

        # Wait for the queue to be processed
        await queue.join()
        pbar_llm.close()

        # Detector
        logger.info(f"Start processing {detector.detector_type} detector")
        pipeline = Pipeline(detector=detector)
        prompts = await prompt_manager.retrieve_prompts_by_detector(detector.detector_type)
        pbar_detector = atqdm(
            total=total_prompts,
            desc=f"{Colors.bright_magenta.value}Processing {detector.detector_type} detector"
        )

        for prompt in prompts:
            await queue.put(
                {"function": pipeline.run_detectors, "args": [],
                 "kwargs": {""'prompt': prompt, "pbar_detector": pbar_detector, "verbose": verbose}}
            )

        # Wait for the queue to be processed
        await queue.join()
        pbar_detector.close()

        prompts = await prompt_manager.retrieve_prompts_by_detector(detector.detector_type)

        tests_passed = sum(1 for test in prompts if test[detector.detector_type].get("status") == "passed")
        tests_failed = sum(1 for test in prompts if test[detector.detector_type].get("status") == "failed")
        tests_skipped = sum(1 for test in prompts if test[detector.detector_type].get("status") == "skipped")

        if tests_passed + tests_skipped + tests_failed != total_prompts:
            logger.error(f"Total test with errors: {total_prompts - tests_passed - tests_skipped - tests_failed}.")

        passed_percentage = _calculate_percentage(tests_passed, total_prompts)
        failed_percentage = _calculate_percentage(tests_failed, total_prompts)
        skipped_percentage = _calculate_percentage(tests_skipped, total_prompts)

        if generate_output_file:
            detector_name = detector.detector_type
            detector_data = {
                "test_data": (
                    f"Total tests: {total_prompts}<br>"
                    f"Passed: {tests_passed} ({passed_percentage:.2f}%)<br>"
                    f"Skipped: {tests_skipped} ({skipped_percentage:.2f}%)<br>"
                    f"Failed: {tests_failed} ({failed_percentage:.2f}%)"
                ),
                "link": f"result_{detector.detector_type}.html",
                "total": total_prompts,
                "passed": tests_passed,
                "skipped": tests_skipped,
                "failed": tests_failed,
                "passed_percentage": f"{passed_percentage:.2f}%",
                "skipped_percentage": f"{skipped_percentage:.2f}%",
                "failed_percentage": f"{failed_percentage:.2f}%",
            }
            output_data[detector_name.capitalize()] = detector_data

        logger.info(
            f"{Colors.bright_cyan.value}Total {detector.detector_type} tests: {total_prompts}\t{Colors.reset.value}"
            f"{Colors.bright_green.value}passed: {tests_passed} ({passed_percentage:.2f}%)\t{Colors.reset.value}"
            f"{Colors.bright_yellow.value}skipped: {tests_skipped} ({skipped_percentage:.2f}%)\t{Colors.reset.value}"
            f"{Colors.bright_red.value}failed: {tests_failed} ({failed_percentage:.2f}%)\n\n{Colors.reset.value}"
        )

        if generate_output_file:
            await report.add_report_data(detector_data, detector_class=detector)
            await report.save_result_to_json(detector)

    if generate_output_file:
        report.generate(debug=debug)

    # Cancel workers
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
